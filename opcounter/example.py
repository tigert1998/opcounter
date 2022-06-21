import torch
import torch.nn as nn
from torchvision.models.resnet import *

from opcounter.counter import counter
from opcounter.hooks import *

if __name__ == "__main__":
    model = resnet18(pretrained=True)
    input_tuple = (torch.randn(1, 3, 224, 224),)
    dst = counter(model, input_tuple, [
        Conv2DForwardHook(),
        LinearForwardHook(),
    ])
    print(dst)
