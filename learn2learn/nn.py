#!/usr/bin/env python3

import torch


class Lambda(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Utility class to create a wrapper based on a lambda function.

    **Arguments**

    * **lmb** (callable) - The function to call in the forward pass.

    **Example**
    ~~~python
    mean23 = Lambda(lambda x: x.mean(dim=[2, 3]))  # mean23 is a Module
    x = features(img)
    x = mean23(x)
    x = x.flatten()
    ~~~
    """

    def __init__(self, lmb):
        super(Lambda, self).__init__()
        self.lmb = lmb

    def forward(self, *args, **kwargs):
        return self.lmb(*args, **kwargs)
