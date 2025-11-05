def get_model(model_name, num_channels, num_classes, img_size, **model_specs):
    if model_name == 'MCFMamba':
        from .mcfmamba import MCFMamba as model
    elif model_name == 'JoiTriNet':
        from .joitrinet import JoiTriNet as model
    else:
        raise NotImplementedError(model_name)
    return model(num_channels, num_classes, img_size, **model_specs)
