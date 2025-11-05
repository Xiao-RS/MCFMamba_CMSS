def get_modeler(modeler_name, modeler_cfg, num_channels, num_classes, img_size):
    if modeler_name == 'Vanilla':
        from .vanilla import Vanilla as modeler
    elif modeler_name == 'Joint':
        from .joint import Joint as modeler
    # elif modeler_name == 'MMOKD':
    #     from .mmokd import MMOKD as modeler
    elif modeler_name == 'MSS':
        from .mss import MSS as modeler
    elif modeler_name == 'Adjust':
        from .adjust import Adjust as modeler
    else:
        raise NotImplementedError(modeler_name)
    return modeler(modeler_cfg, num_channels, num_classes, img_size)
