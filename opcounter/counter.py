def counter(model, input_tuple, hooks):
    dst = {}

    handles = []

    for module in model.modules():
        for hook in hooks:
            if isinstance(module, hook.module):
                handle = module.register_forward_hook(hook(dst))
                handles.append(handle)

    training = model.training
    model.eval()
    model(*input_tuple)

    for handle in handles:
        handle.remove()

    model.train(training)
    return dst
