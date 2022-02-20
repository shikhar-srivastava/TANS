from functools import reduce
from typing import Union

import torch
from torch import nn


def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


if __name__ == '__main__':
    from torchvision.models import resnet34
    
    model = resnet34()
    get_module_by_name(model, 'layer1.0.relu')