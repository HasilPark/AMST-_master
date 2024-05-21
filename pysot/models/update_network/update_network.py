from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch as t

class UpdateNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpdateNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 1)
        )

        for modules in [self.update]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)

    def forward(self, x):
        x = self.update(x)
        return x

class Update_feat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Update_feat, self).__init__()
        self.num = len(out_channels)
        for i in range(self.num):
            self.add_module('update'+str(i+3),
                            UpdateNet(in_channels[i], out_channels[i]))

    def forward(self, zf, uf, z0):
        out = []
        for i in range(self.num):
            concat_f = t.cat((zf[i], uf[i]), dim=1)
            update_layer = getattr(self, 'update'+str(i+3))
            out.append(update_layer(concat_f) + z0[i])
        if not t.isfinite(out[0].sum()):
            print("update stop")
        return out
