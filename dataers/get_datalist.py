import os
import glob
from random import shuffle
from engine.utils import log_msg


def get_datalist(dataset_name="DFC2020", repartition=True, trainval_rate=0.8, train_rate=None):
    if train_rate is None:
        datalist_root = os.path.join(f"../DataList/{dataset_name}", str(trainval_rate))
    else:
        datalist_root = os.path.join(f"../DataList/{dataset_name}", str(trainval_rate), str(train_rate))

    if repartition:
        print(log_msg("repartition ...", "PROCESS"), end=" ")

        if dataset_name == "DFC2020":
            lbl_list_dict = {}
            dataroot = os.path.abspath('/Data/DFC2020')
            lbl_list = glob.glob(os.path.join(dataroot, "*/dfc_*/*.tif"))
            shuffle(lbl_list)
            trainval_num = int(len(lbl_list) * trainval_rate)
            lbl_list_dict["test"] = lbl_list[trainval_num:]
            for r in [0.1, 0.3, 0.6, 1.0]:
                train_num = int(trainval_num * r)
                lbl_list_dict["train"] = lbl_list[0:train_num]
                lbl_list_dict["val"] = lbl_list[train_num:trainval_num]
                datalist_root_temp = os.path.join(datalist_root, str(r))
                if not os.path.exists(datalist_root_temp):
                    os.makedirs(datalist_root_temp)
                patterns = ["train", "val", "test"]
                for pattern in patterns:
                    txt = open(os.path.join(datalist_root_temp, '{}_list.txt'.format(pattern)), 'w')
                    txt.write("id,lbl,opt,sar\n")
                    for lbl_path in lbl_list_dict[pattern]:
                        patch_ids = os.path.basename(lbl_path).split(".")[0]
                        opt_path = lbl_path.replace("dfc", "s2")
                        dsm_path = lbl_path.replace("dfc", "s1")
                        txt.write('{},{},{},{}\n'.format(patch_ids, lbl_path, opt_path, dsm_path))
                    txt.close()
        else:
            raise NotImplementedError(dataset_name)

        print("DataList file saved successfully")

    return datalist_root, train_rate if train_rate == 1.0 else trainval_rate