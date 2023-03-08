# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
from functools import partial
import copy
import numpy as np
import torch
import torch.nn.functional as F


class Conv1o1Layer(torch.nn.Module):
    def __init__(self, weights):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)

    def forward(self, x):
        weight = self.weight
        xnorm = torch.norm(x, dim=1, keepdim=True)
        boo_zero = (xnorm == 0).type(torch.FloatTensor).to('cuda:0')
        xnorm = xnorm + boo_zero
        xn = x / xnorm
        wnorm = torch.norm(weight, dim=1, keepdim=True)
        weightnorm2 = weight / wnorm
        out = F.conv1d(xn, weightnorm2)
        if torch.sum(torch.isnan(out)) > 0:
            print('isnan conv1o1')
        return out


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
        use_new=True,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_new:
            categories = {
                "bed": 0,
                "table": 1,
                "sofa": 2,
                "chair": 3,
                "toilet": 4,
                "desk": 5,
                "dresser": 6,
                "night_stand": 7,
                "bookshelf": 8,
                "bathtub": 9,
            }
            vc_weights = []
            for cat in categories:
                vc = np.load('./features/' + cat + '/center_dict_'+str(7)+'.pickle', allow_pickle=True)
                # print('VC', vc.shape)
                vc_weights.append(vc)
            vc_weights = np.array(vc_weights)
            vc_weights = np.reshape(vc_weights, (-1,256))
            vc_weights = torch.from_numpy(np.array(vc_weights)).to(torch.float32).cuda()

            layer = Conv1o1Layer(vc_weights[:,:,np.newaxis])
            layers.append(layer)

        if use_new:
            new_dim = 100
        else:
            new_dim = prev_dim
        if use_conv:
            layer = nn.Conv1d(new_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(new_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
