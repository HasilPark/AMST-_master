import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.utile_tctrack.trantime_encoder import Transformer_time_encoder
from pysot.models.utile_tctrack.transaptio_encoder import Tranformer_spatio_encoder
from pysot.models.utile_tctrack.aggregation_encoder import Transformer_aggregation_encoder
from pysot.models.utile_tctrack.share_decoder import Share_decoder

class TCTtest(nn.Module):

    def __init__(self, cfg):
        super(TCTtest, self).__init__()

        self.conv2_spatio = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv1_spatio = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv2_time = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv1_time = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        channel = 192

        self.convloc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 4, kernel_size=3, stride=1, padding=1),
        )

        self.convcls = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(inplace=True),
        )

        self.transformer_time_encoder = Transformer_time_encoder(channel, 6, 1)
        self.transformer_spatio_encoder = Tranformer_spatio_encoder(channel, 6, 1)
        self.transformer_aggregation_encoder = Transformer_aggregation_encoder(channel, 6, 1)
        self.share_decoder = Share_decoder(channel, 6, 2)

        self.row_embed = nn.Embedding(50, channel // 2)
        self.col_embed = nn.Embedding(50, channel // 2)
        self.reset_parameters()

        self.cls1 = nn.Conv2d(channel // 8, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channel // 8, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.conv1_time, self.conv2_time, self.conv1_spatio, self.conv2_spatio, self.convloc,
                        self.convcls, self.cls1, self.cls2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                # t.nn.init.constant_(l.bias, 0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

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

    def forward(self, x, z, ppres):

        res3 = self.conv2_time(self.xcorr_depthwise(x[2], z[2]))

        b, c, w, h = res3.size()
        memory_time = self.transformer_time_encoder((ppres).view(b, c, -1).permute(2, 0, 1), \
                                                    res3.view(b, c, -1).permute(2, 0, 1))

        res1 = self.conv1_spatio(self.xcorr_depthwise(x[0], z[0]))
        res2 = self.conv2_spatio(self.xcorr_depthwise(x[1], z[1]))

        h, w = res3.shape[-2:]
        i = t.arange(w).cuda()
        j = t.arange(h).cuda()

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = t.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res3.shape[0], 1, 1, 1)

        b, c, w, h = res3.size()

        memory_spatio = self.transformer_spatio_encoder((pos + res1).view(b, c, -1).permute(2, 0, 1),
                                                        (pos + res2).view(b, c, -1).permute(2, 0, 1))

        concat_memory = self.transformer_aggregation_encoder(memory_time, memory_spatio)

        res = self.share_decoder(concat_memory.view(b, c, -1).permute(2, 0, 1), memory_time.view(b, c, -1).permute(2, 0, 1), memory_spatio.view(b, c, -1).permute(2, 0, 1),
                                 res3.view(b, c, -1).permute(2, 0, 1))

        res = res.permute(1, 2, 0).view(b, c, w, h)

        loc = self.convloc(res)
        acls = self.convcls(res)

        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)

        return loc, cls1, cls2, memory_time





