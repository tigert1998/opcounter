import torch.nn as nn

import numpy as np


class Conv2DForwardHook:
    module = nn.Conv2d

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, nn.Conv2d)
            assert module.groups == 1 and module.dilation == (1, 1)
            b, _, h, w = input_tensors[0].shape
            out_h = (h + 2 * module.padding[0] - module.kernel_size[0]) \
                // module.stride[0] + 1
            out_w = (w + 2 * module.padding[1] - module.kernel_size[1]) \
                // module.stride[1] + 1
            muladds = b * module.out_channels * out_h * out_w * \
                module.kernel_size[0] * \
                module.kernel_size[1] * module.in_channels
            mem = 4 * (b * module.in_channels * h * w + np.prod(module.weight.shape) +
                       b * module.out_channels * out_h * out_w)
            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["mem"] = dst.get("mem", 0) + mem

        return func


class LinearForwardHook:
    module = nn.Linear

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, nn.Linear)
            in_features, out_features = module.in_features, module.out_features
            b, _ = input_tensors[0].shape
            muladds = b * in_features * out_features
            mem = 4 * (b * in_features + b * out_features +
                       in_features * out_features)
            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["mem"] = dst.get("mem", 0) + mem
        return func
