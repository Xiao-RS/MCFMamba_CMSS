import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import torch
from dataers import get_dataloaders
from models import get_model
from modelers import get_modeler
from engine import get_trainer, get_tester
from engine.base_config import CFG
from engine.utils import random_seed, log_msg, load_cpt


def main_train(cfg, resume):
    random_seed(seed_value=cfg.EXPERIMENT.SEED_VALUE, deter=cfg.EXPERIMENT.ALGO_DETER)

    print(log_msg("Dump cfg", "PROCESS"))
    log_path = os.path.join(cfg.SOLVER.LOG_PREFIX, cfg.EXPERIMENT.PROJECT, cfg.EXPERIMENT.NAME, cfg.EXPERIMENT.TAG)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not resume:
        with open(os.path.join(log_path, "log_INFO.yaml"), "a") as writer:
            yaml.dump(cfg, writer)

    print(log_msg("Loading Dataloader", "PROCESS"))
    train_loader, val_loader, labels_dict, num_channels, img_size = get_dataloaders(cfg.DATASET, pattern="train_val")
    num_classes = len(labels_dict)

    print(log_msg("Loading Modeler", "PROCESS"))
    modeler = get_modeler(
        cfg.MODELER.TYPE, cfg.MODELER,
        num_channels, num_classes, img_size,
    )
    modeler = torch.nn.DataParallel(modeler.cuda())

    print(log_msg("Loading Trainer", "PROCESS"))
    trainer = get_trainer(
        cfg.SOLVER,
        log_path, train_loader, val_loader, modeler, num_classes, labels_dict
    )

    trainer.train(resume=resume)


def main_test(cfg, xlsx_name=None):
    log_path = os.path.join(cfg.SOLVER.LOG_PREFIX, cfg.EXPERIMENT.PROJECT, cfg.EXPERIMENT.NAME, cfg.EXPERIMENT.TAG)

    print(log_msg("Loading Dataloader", "PROCESS"))
    test_loader, labels_dict, num_channels, img_size = get_dataloaders(cfg.DATASET, pattern="test")
    num_classes = len(labels_dict)

    print(log_msg("Loading Model", "PROCESS"))
    model = get_model(
        cfg.MODELER.MODEL,
        num_channels, num_classes, img_size, **cfg.MODELER.MODEL_SPECS[0]
    )

    best_ckpt = os.path.join(log_path, "cpt", "model_best.pth")
    model.load_state_dict(load_cpt(best_ckpt)["model"])

    print(log_msg("Loading Tester", "PROCESS"))
    tester = get_tester(
        cfg.SOLVER.TESTER,
        log_path, test_loader, model.cuda(), num_classes, labels_dict
    )

    if xlsx_name is None:
        resultxlsx = os.path.join(log_path, "Best.xlsx")
    else:
        resultxlsx = os.path.join(
            cfg.SOLVER.LOG_PREFIX, cfg.EXPERIMENT.PROJECT, cfg.EXPERIMENT.NAME, f"{xlsx_name}.xlsx"
        )
    tester.test(resultxlsx)


if __name__ == "__main__":
    # train MSS
    cfg = CFG.clone()
    f_yaml = r"E:\Code\MCFMambaCMSS\configs\DFC2020\MSS.yaml"
    cfg.merge_from_file(f_yaml)
    cfg.defrost()
    cfg.freeze()
    main_train(cfg, resume=False)

    NAME_opts = {
        '0.1': [],
        '0.3': [
            "EXPERIMENT.NAME", "Fusion-0.3",
            "DATASET.SPLIT_SPECS", [{'repartition': False, 'trainval_rate': 0.8, 'train_rate': 0.3, }],
        ],
        '0.6': [
            "EXPERIMENT.NAME", "Fusion-0.6",
            "DATASET.SPLIT_SPECS", [{'repartition': False, 'trainval_rate': 0.8, 'train_rate': 0.6, }],
        ],
        '1.0': [
            "EXPERIMENT.NAME", "Fusion-1.0",
            "DATASET.SPLIT_SPECS", [{'repartition': False, 'trainval_rate': 0.8, 'train_rate': 1.0, }],
            "SOLVER.TRAINER", "NoVal",
            "SOLVER.SAVE_CPT_FREQ", 1,
        ],
    }
    for i in ['0.1', '0.3', '0.6', '1.0']:

        # train adjust
        cfg = CFG.clone()
        f_yaml = r'E:\Code\MCFMambaCMSS\configs\DFC2020\Adjust-0.1.yaml'
        cfg.merge_from_file(f_yaml)
        cfg.defrost()
        cfg.merge_from_list(NAME_opts[i])
        cfg.freeze()
        main_train(cfg, resume=False)
        main_test(cfg, 'MCFMamba+CMSS')

