from pysot.models.utile_tctrack.flops_net import ModelBuilder_tctrack

from thop import profile
from thop.utils import clever_format
import torch
from pysot.core.config import cfg
from torchsummaryX import summary
import argparse
from torchinfo import summary
# from efficientnet_pytorch.utils import Conv2dDynamicSamePadding
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding
# from efficientnet_pytorch.utils import MemoryEfficientSwish
# from thop.vision.basic_hooks import count_convNd, zero_ops

if __name__ == "__main__":
    # Compute the Flops and Params of our LightTrack-Mobile model
    # build the searched model
    parser = argparse.ArgumentParser(description='AMST_Square tracking')
    parser.add_argument('--cfg', type=str, default='../experiments/AMST_Square/config.yaml',
                        help='configuration of tracking')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='compulsory for pytorch launcer')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg.merge_from_file(args.cfg)

    model = ModelBuilder_tctrack('train').to(device)

    print(model)

    px = torch.randn(1, 1, 3, 287, 287)
    xf = torch.randn(1, 3, 287, 287)
    uf = torch.randn(1, 3, 127, 127)
    zf = torch.randn(1, 3, 127, 127)

    #######backbone
    # summary(model.backbone, input_size=[(1, 1, 3, 287, 287)])

    summary(model, input_size=[(1, 3, 127, 127), (1, 3, 127, 127), (1, 1, 3, 287, 287), (1, 3, 287, 287)])
    print("vgg112 : ", sum(p.numel() for p in model.grader.convcls.parameters() if p.requires_grad))
    # for parameter in model.parameters():
    #    if parameter.requires_grad:
    #        aa = torch.numel(parameter)
    #        print(aa)

    inp = {'cls': torch.randn(1, 128, 16, 16), 'reg': torch.randn(1, 128, 16, 16)}

    # oup = model(zf, uf, px, xf)

    # custom_ops = {
    #     Conv2dDynamicSamePadding: count_convNd,
    #     Conv2dStaticSamePadding: count_convNd,
    #     MemoryEfficientSwish: zero_ops,
    # }
    # compute FLOPs and Params
    # the whole model
    macs, params = profile(model, inputs=(zf, uf, px, xf), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    # backbone
    # macs, params = profile(backbone, inputs=(x,), custom_ops=None, verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")
    # print('backbone macs is ', macs)
    # print('backbone params is ', params)
    # # head
    # macs, params = profile(head, inputs=(inp,), verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")
    # print('head macs is ', macs)
    # print('head params is ', params)
    print("vgg11 : ", sum(p.numel() for p in model.grader.conv1_time.parameters() if p.requires_grad))
    print("vgg11 : ", sum(p.numel() for p in model.grader.conv2_time.parameters() if p.requires_grad))
    print("vgg11 : ", sum(p.numel() for p in model.grader.conv1_spatio.parameters() if p.requires_grad))
    print("vgg11 : ", sum(p.numel() for p in model.grader.conv2_spatio.parameters() if p.requires_grad))
