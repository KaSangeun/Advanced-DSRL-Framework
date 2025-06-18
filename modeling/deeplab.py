import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.sr_decoder import build_sr_decoder

class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

        # self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False): # output_stride=16 is in DSRL paper 
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm) # encoder
        self.aspp = build_aspp(backbone, output_stride, BatchNorm) # encoder
        self.decoder = build_decoder(num_classes, backbone, BatchNorm) # ss decoder
        self.sr_decoder = build_sr_decoder(num_classes,backbone,BatchNorm) # sr decoder 
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes,3,1),
            torch.nn.BatchNorm2d(3),  # add BN layer 
            torch.nn.ReLU(inplace=True)
        )

        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        self.up_edsr_1 = EDSRConv(64,64)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        self.up_edsr_2 = EDSRConv(32,32)
        self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        self.up_edsr_3 = EDSRConv(16,16)
        self.up_conv_last = nn.Conv2d(16,3,1)

        self.freeze_bn = freeze_bn

    def forward(self, input): # input image (2,3, 512, 512), SSSR & SISR Branch in DSRL framework (before FA Loss)
        # x: deeper feature, low_level_feat: lower feature(e.g. edge, texture)
        x, low_level_feat = self.backbone(input) # encoder, run resnet forward() (x: 32 x 32 x 2048, low_level_feat: 128 x 128 x 256)
        x = self.aspp(x) # encoder (atrous conv) (x: 32 x 32 x 256)
        x_seg = self.decoder(x, low_level_feat) # SSSR, share with encoder (x_seg: 128 x 128 x num_classes)
        x_sr= self.sr_decoder(x, low_level_feat) # SISR, share with encoder (x_sr: 128 x 128 x 64)
        """ x_seg_up: feature1 for cross-attnetion map, x_seg_up2: output of SSSR branch """
        x_seg_up = F.interpolate(x_seg, size=input.size()[2:], mode='bicubic', align_corners=True) # 512 x 512 x num_classes (changed bilinear to bicubic)
        x_seg_up2 = F.interpolate(x_seg_up,size=[2*i for i in input.size()[2:]], mode='bicubic', align_corners=True) # 1024 x 1024 x num_classes (changed bilinear to bicubic)
        
        # upsampling x_sr part 
        x_sr_up = self.up_sr_1(x_sr) # 256 x 256 x 64
        x_sr_up=self.up_edsr_1(x_sr_up) # 256 x 256 x 64

        x_sr_up = self.up_sr_2(x_sr_up) # 512 x 512 x 32
        x_sr_up=self.up_edsr_2(x_sr_up) # 512 x 512 x 32

        """ x_sr_up: feature2 for cross-attention map, x_sr_up2: output of SISR branch """

        x_sr_up2 = self.up_sr_3(x_sr_up) # 1024 x 1024 x 16
        x_sr_up2=self.up_edsr_3(x_sr_up2) # 1024 x 1024 x 16
        x_sr_up2=self.up_conv_last(x_sr_up2) # 1024 x 1024 x 3 (3: RGB image) --> SISR IMAGE 

        #return x_seg_up, x_sr_up, self.pointwise(x_seg_up), x_sr_up # ss output, sr output, ss feature(1024 x 1024 x 3), sr feature 
        return x_seg_up2, x_sr_up2, x_seg_up, x_sr_up 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


