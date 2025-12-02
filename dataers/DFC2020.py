import os
import random
import rasterio
import numpy as np
import pandas as pd
from enum import Enum
from torch.utils.data import Dataset
from engine.utils import log_msg


class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []
    Random3 = random.choice([[VV, VH, VV], [VV, VH, VH]])
    COPY4 = [VV, VH, VV, VH]


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    HR = [B02, B03, B04, B08]
    MR = [B05, B06, B07, B08A, B11, B12]
    LR = [B01, B09, B10]
    NONE = []
    RGB = [B04, B03, B02]
    RGBN = [B04, B02, B03, B08]


S1Bands_composition = {"ALL": S1Bands.ALL.value, "NONE": S1Bands.NONE.value,
                       "Random3": S1Bands.Random3.value, "COPY4": S1Bands.COPY4.value}
S2Bands_composition = {"ALL": S2Bands.ALL.value, "HR": S2Bands.HR.value, "MR": S2Bands.MR.value, "LR": S2Bands.LR.value,
                       "NONE": S2Bands.NONE.value, "RGB": S2Bands.RGB.value, "RGBN": S2Bands.RGBN.value, }
S2_MEAN = np.array([1254.019, 1013.218, 919.146, 763.869, 985.786, 1670.818, 1961.507,
                    1912.286, 2131.904, 698.649, 11.146, 1449.988, 892.639])
S2_STD = np.array([80.979, 141.464, 172.787, 247.855, 236.570, 382.324, 478.105,
                   507.345, 528.757, 142.551, 1.765, 425.619, 340.500])
S1_MEAN = np.array([-12.549, -18.553])
S1_STD = np.array([2.345, 1.987])

DFC2020_CLASSES = [0,1,1,1,1,1,2,2,3,3,4,5,6,7,6,8,9,10]


class DFC2020(Dataset):
    def __init__(self, pattern='val', datalist_root='./../DataList/DFC2020', transform=None,
                 modality={'train': 'OPT_SAR', 'val': 'OPT_SAR', 'test': 'OPT_SAR'},
                 OPTBands_load="RGBN", SARBands_load="ALL", **kwargs):
        super().__init__()
        assert pattern in ["val", "train", "test"]
        self.df = pd.read_csv(os.path.join(datalist_root, pattern + '_list.txt'))
        self.transform = transform
        self.modality = modality[pattern]

        self.labels_dict = {
            'Forests': {0: (0, 100, 0)}, 'Shrublands': {1: (255, 187, 34)}, 'Grasslands': {2: (100, 255, 0)},
            'Wetlands': {3: (0, 150, 160)}, 'Croplands': {4: (250, 160, 255)}, 'Urban/Built-up': {5: (255, 0, 0)},
            'Barren': {6: (190, 190, 190)}, 'Water': {7: (0, 0, 255)},
        }
        self.n_classes = len(self.labels_dict)
        self.img_size = (256, 256)
        self.OPTBands = S2Bands_composition[OPTBands_load]
        self.SARBands = S1Bands_composition[SARBands_load]
        if self.modality == "OPT_SAR":
            self.n_channels = {'m1': len(self.OPTBands), 'm2': len(self.SARBands), }
        elif self.modality == "OPT_SAR_MSS":
            self.n_channels = {'m1': len(self.OPTBands), 'm2': len(self.SARBands), }
            from .get_mask import MaskGenerator
            self.mask_generator = MaskGenerator(
                input_size=self.img_size,
                mask_patch_size=kwargs['mask_patch_size'],
                model_patch_size=kwargs['model_patch_size'],
                mask_ratio=kwargs['mask_ratio'],
                mask_type=kwargs['mask_type'],
                strategy=kwargs['mask_strategy'],
            )
        else:
            raise ValueError("modality must be one of 'OPT_SAR', OPT_SAR_MSS")

        print(log_msg(pattern + f" Images: {len(self.df)}", "INFO"))

    def _trans(self, sample):
        if self.transform:
            return self.transform(**sample)
        else:
            return sample

    def _load_opt(self, sample_loc):
        with rasterio.open(sample_loc["opt"]) as data:
            s2 = data.read(self.OPTBands)
        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
        return s2

    def _load_sar(self, sample_loc):
        with rasterio.open(sample_loc["sar"]) as data:
            s1 = data.read(self.SARBands)
        s1 = s1.astype(np.float32)
        s1 = np.nan_to_num(s1)
        s1 = np.clip(s1, -25, 0)
        s1 /= 25
        s1 += 1
        return s1

    def _load_lbl(self, sample_loc):
        with rasterio.open(sample_loc["lbl"]) as data:
            dfc = data.read()
        dfc = dfc.astype(np.int64).squeeze()
        dfc[(dfc == 3) | (dfc == 8)] = 0
        dfc[(dfc > 3) & (dfc < 8)] -= 1
        dfc[dfc > 8] -= 2
        dfc -= 1
        # set ignore mask
        dfc[dfc == -1] = -100
        return dfc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_loc = self.df.iloc[idx]
        if self.modality == "OPT_SAR_MSS":
            opt = self._load_opt(sample_loc)
            sar = self._load_sar(sample_loc)
            _spl = self._trans({'image': opt, 'image_sar': sar, })

            mask1, mask2 = self.mask_generator()
            
            sample = {
                'id': sample_loc["id"],
                'image': {'m1': _spl['image'], 'm2': _spl['image_sar'], 'mask1': mask1, 'mask2': mask2},
                'label': 0,
            }
            return sample
        
        lbl = self._load_lbl(sample_loc)
        if self.modality == "OPT_SAR":
            opt = self._load_opt(sample_loc)
            sar = self._load_sar(sample_loc)
            _spl = self._trans({'image': opt, 'mask': lbl, 'image_sar': sar, })

            sample = {
                'id': sample_loc["id"],
                'image': {'m1': _spl['image'], 'm2': _spl['image_sar']},
                'label': _spl['mask'],
            }
            return sample
