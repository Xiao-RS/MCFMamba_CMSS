import numpy as np


class MaskGenerator:
    def __init__(self, input_size=[256, 256], mask_patch_size=32, model_patch_size=4,
                 mask_ratio=0.6, strategy='comp'):

        self.input_size = np.array(input_size)
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size[0] % self.mask_patch_size == 0
        assert self.input_size[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.maskpatch_count = self.rand_size[0] * self.rand_size[1]
        self.masked_count = int(np.ceil(self.maskpatch_count * self.mask_ratio))
        self.scale = self.mask_patch_size // self.model_patch_size

        # 根据策略选择掩码生成策略
        if strategy == 'comp':
            self.strategy = self.gen_comp_masks
        elif strategy == 'indiv':
            self.strategy = self.gen_indiv_masks
        else:
            raise AssertionError("Not valid strategy!")

    def gen_mask(self):
        mask_idx = np.random.permutation(self.maskpatch_count)[:self.masked_count]
        mask = np.zeros(self.maskpatch_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = np.expand_dims(mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1), axis=0)

        return mask


    def gen_comp_masks(self):
        mask = self.gen_mask()
        return mask, 1 - mask

    def gen_indiv_masks(self):
        mask1 = self.gen_mask()
        mask2 = self.gen_mask()
        return mask1, mask2

    def __call__(self):
        return self.strategy()
