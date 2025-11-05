def get_trainer(cfg_SOLVER, log_path, train_loader, val_loader, modeler, num_classes, labels_dict):
    if cfg_SOLVER.TRAINER == 'Base':
        from .BaseSolvers import Trainer as trainer
    elif cfg_SOLVER.TRAINER == 'NoVal':
        from .Trainer_NoVal import NoValTrainer as trainer
    else:
        NotImplementedError(cfg_SOLVER.TRAINER)
    return trainer(
        cfg_SOLVER, log_path, train_loader, val_loader, modeler, num_classes, labels_dict
    )


def get_tester(tester_name, log_path, test_loader, model, num_classes, labels_dict):
    if tester_name == 'Base':
        from .BaseSolvers import Tester as tester
    else:
        NotImplementedError(tester_name)
    return tester(
        log_path, test_loader, model.cuda(), num_classes, labels_dict
    )
