import torch.nn as nn


def counter(model, input_tuple, hooks):
    dst = {}

    handles = []

    def dfs(module: nn.Module):
        for hook in hooks:
            if isinstance(module, hook.module):
                handle = module.register_forward_hook(hook(dst))
                handles.append(handle)
                return
        for module in module.children():
            dfs(module)

    dfs(model)

    training = model.training
    model.eval()
    model(*input_tuple)

    for handle in handles:
        handle.remove()

    model.train(training)
    return dst
