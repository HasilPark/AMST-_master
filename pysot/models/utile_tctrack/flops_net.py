# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.utile_tctrack.loss import select_cross_entropy_loss, weight_l1_loss, CIOULoss
from pysot.models.backbone.temporalbackbone import TemporalAlexNet

from pysot.models.utile_tctrack.utile import TcT
from pysot.models.utile_tctrack.utiletest import TCTtest
from pysot.models.update_network import get_update_feat

import numpy as np


class ModelBuilder_tctrack(nn.Module):
    def __init__(self, label):
        super(ModelBuilder_tctrack, self).__init__()

        self.backbone = TemporalAlexNet().cuda()
        # print("vgg11 : ", sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))
        #
        self.updatenet = get_update_feat(**cfg.UPDATE.KWARGS).cuda()

        # print("vgg11 : ", sum(p.numel() for p in self.updatenet.parameters() if p.requires_grad))

        if label == 'test':
            self.grader = TCTtest(cfg).cuda()
        else:
            self.grader = TcT(cfg).cuda()

        # print("vgg11 : ", sum(p.numel() for p in self.grader.parameters() if p.requires_grad))

        self.cls3loss = nn.BCEWithLogitsLoss()
        self.IOULOSS = CIOULoss()

    def template(self, z, x):
        with t.no_grad():
            zf, _, _, _ = self.backbone.init(z)

            xf, xfeat1, xfeat2, xfeat3 = self.backbone.init(x)

            ppres = self.grader.conv1_time(self.xcorr_depthwise(xf[2], zf[2]))

            self.zf = zf
            self.zf0 = zf
            self.memory = ppres
            self.featset1 = xfeat1
            self.featset2 = xfeat2
            self.featset3 = xfeat3

    def templete_update(self, z):
        with t.no_grad():
            uf, _, _, _ = self.backbone.init(z)

            update_f = self.updatenet(self.zf, uf, self.zf0)

            self.zf = update_f

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def track(self, x):
        with t.no_grad():
            xf, xfeat1, xfeat2, xfeat3 = self.backbone.eachtest(x, self.featset1, self.featset2, self.featset3)

            loc, cls2, cls3, memory = self.grader(xf, self.zf, self.memory)

            self.memory = memory
            self.featset1 = xfeat1
            self.featset2 = xfeat2
            self.featset3 = xfeat3

        return {
            'cls2': cls2,
            'cls3': cls3,
            'loc': loc
        }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls

    def getcentercuda(self, mapp):

        def dcon(x):
            x[t.where(x <= -1)] = -0.99
            x[t.where(x >= 1)] = 0.99
            return (t.log(1 + x) - t.log(1 - x)) / 2

        size = mapp.size()[3]
        # location
        x = t.Tensor(np.tile((16 * (np.linspace(0, size - 1, size)) + 63) - 287 // 2, size).reshape(-1)).cuda()
        y = t.Tensor(
            np.tile((16 * (np.linspace(0, size - 1, size)) + 63).reshape(-1, 1) - 287 // 2, size).reshape(-1)).cuda()

        shap = dcon(mapp) * 143

        xx = np.int16(np.tile(np.linspace(0, size - 1, size), size).reshape(-1))
        yy = np.int16(np.tile(np.linspace(0, size - 1, size).reshape(-1, 1), size).reshape(-1))

        w = shap[:, 0, yy, xx] + shap[:, 1, yy, xx]
        h = shap[:, 2, yy, xx] + shap[:, 3, yy, xx]
        x = x - shap[:, 0, yy, xx] + w / 2 + 287 // 2
        y = y - shap[:, 2, yy, xx] + h / 2 + 287 // 2

        anchor = t.zeros((cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPU, size ** 2, 4)).cuda()

        anchor[:, :, 0] = x - w / 2
        anchor[:, :, 1] = y - h / 2
        anchor[:, :, 2] = x + w / 2
        anchor[:, :, 3] = y + h / 2
        return anchor

    def forward(self, template, update, presearch, search):
        """ only used in training
        """
        presearch = presearch.cuda()
        template = template.cuda()
        update = update.cuda()
        search = search.cuda()

        presearch = t.cat((presearch[:, cfg.TRAIN.videorangemax - 3:, :, :, :], search.unsqueeze(1)), 1)

        zf = self.backbone(template.unsqueeze(1))
        uf = self.backbone(update.unsqueeze(1))
        xf = self.backbone(presearch)  ###b l c w h

        update_f = self.updatenet(zf, uf, zf)

        for i in range(len(xf)):
            xf[i] = xf[i].view(1, 1 + 1, xf[i].size(-3), xf[i].size(-2), xf[i].size(-1))

        loc_o, cls2_o, cls3_o = self.grader(xf, update_f)


        return loc_o

