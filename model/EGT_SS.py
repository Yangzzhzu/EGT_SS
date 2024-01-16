from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from timm.models.layers import to_2tuple
from model.dilated_resnet import get_resnet101_baseline

class CLTRF(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
 
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
 
    def forward(self, x):
        B, C, H, W = x.shape
 
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
 
        return x, (H, W)

class EDRF(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(EDRF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class sundry(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(sundry, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return x + out

class EGT(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(EGT, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return x + out

class EGT_SS(nn.Module):

    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d, layers=101, multi_grid=(1, 1, 1), criterion=None,
                 pretrained=True):

        super(EGT_SS, self).__init__()
        self.criterion = criterion
        self.num_classes = num_classes
        self.BatchNorm = BatchNorm
        if layers == 101:
            resnet = get_resnet101_baseline(pretrained=pretrained, num_classes=num_classes, BatchNorm=BatchNorm,
                                            multi_grid=multi_grid)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.interpolate = F.interpolate
        del resnet

        self.EDRF1 = EDRF(256,256)
        self.EDRF2 = EDRF(512,256)
        self.EDRF3 = EDRF(1024,256)
        self.EDRF4 = EDRF(2048,256)

        self.down = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
        )

        self.sundry = sundry(2048, 256, 256, BatchNorm)

        self.final_seg = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
        self.aux_seg = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
    
    def edge_out(self):
        return nn.Sequential(nn.Conv2d(256, 1, 1), nn.Sigmoid())

    def forward(self, inp, gts=None):

        x_size = inp.size()

        m0 = self.layer0(inp)
        m0_size = m0.size()
        ma = self.CLTRF(x_size[2],2,x_size[1])
        m0 = F.interpolate(m0, size=(ma.size()[2], ma.size()[3]), mode='bilinear', align_corners=True)
        m0 = m0 + ma
        m0 = torch.cat((m0, ma), dim=1)
        
        m1 = self.layer1(m0)
        m1_size = m1.size()
        mb = self.CLTRF(m0_size[2],4,m0_size[1])
        m1 = F.interpolate(m1, size=(mb.size()[2], mb.size()[3]), mode='bilinear', align_corners=True)
        m1 = m1 + ma
        m1 = torch.cat((m1, mb), dim=1)
   
        m2 = self.layer2(m1)
        m2_size = m2.size()
        mc = self.CLTRF(m1_size[2],8,m1_size[1])
        m2 = F.interpolate(m2, size=(mc.size()[2], mc.size()[3]), mode='bilinear', align_corners=True)
        m2 = m2 + ma
        m2 = torch.cat((m2, mc), dim=1)

        m3 = self.layer3(m2)
        m3_size = m3.size()
        md = self.CLTRF(m2_size[2],16,m2_size[1])
        m3 = F.interpolate(m3, size=(md.size()[2], md.size()[3]), mode='bilinear', align_corners=True)
        m3 = m3 + ma
        m3 = torch.cat((m3, md), dim=1)

        m4 = self.layer4(m3)
        me = self.CLTRF(m3_size[2],16,m3_size[1])
        m4 = F.interpolate(m4, size=(me.size()[2], me.size()[3]), mode='bilinear', align_corners=True)
        m4 = m4 + ma
        m4 = torch.cat((m4, me), dim=1)


        e1 = self.EDRF1(m1)
        e2 = self.EDRF1(m2)
        e3 = self.EDRF1(m3)
        e4 = self.EDRF1(m4)

        e1 = self.interpolate(e1, m4.size()[2:], mode='bilinear', align_corners=True)
        e2 = self.interpolate(e2, m4.size()[2:], mode='bilinear', align_corners=True)
        e3 = self.interpolate(e3, m4.size()[2:], mode='bilinear', align_corners=True)
        e4 = self.interpolate(e4, m4.size()[2:], mode='bilinear', align_corners=True)

        e = torch.cat((e1, e2, e3, e4), dim=1)
        e = self.down(e)
        
        out_feature = self.sundry(m4, e)

        seg_out_ = self.final_seg(out_feature)
        seg_out = self.interpolate(seg_out_, x_size[2:], mode='bilinear', align_corners=True)
       
        if self.training:
            aux_seg_out = self.interpolate(self.aux_seg(m3), x_size[2:], mode='bilinear', align_corners=True)
            return seg_out, self.criterion((seg_out,aux_seg_out), gts)
        else:
            return seg_out, 

