import torch
import torch.nn as nn
from einops import rearrange

def Conv3x3BN(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )


def Conv3x3BNActivation(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1x1BNActivation(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )


class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(MV2Block, self).__init__()
        self.stride = stride
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNActivation(in_channels, mid_channels),
            Conv3x3BNActivation(mid_channels, mid_channels, stride, groups=mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MBTRBlock(nn.Module):
    def __init__(self, dim, depth, channel, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv3x3BNActivation(channel, channel)
        self.conv2 = Conv1x1BNActivation(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = Conv1x1BNActivation(dim, channel)
        self.conv4 = Conv3x3BNActivation(2 * channel, channel)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MBTR(nn.Module):
    def __init__(self, dims, channels, expansion=4, patch_size=(2, 2), num_classes=1000):
        super(MBTR, self).__init__()
        depth = [2, 4, 3]

        self.conv1 = Conv3x3BNActivation(3, channels[0], 2)
        self.layer1 = MV2Block(in_channels=channels[0], out_channels=channels[1], stride=1, expansion_factor=expansion)

        self.layer2 = nn.Sequential(
            MV2Block(in_channels=channels[1], out_channels=channels[2], stride=2, expansion_factor=expansion),
            MV2Block(in_channels=channels[2], out_channels=channels[3], stride=1, expansion_factor=expansion),
            MV2Block(in_channels=channels[3], out_channels=channels[3], stride=1, expansion_factor=expansion)
        )

        self.layer3 = nn.Sequential(
            MV2Block(in_channels=channels[3], out_channels=channels[4], stride=2, expansion_factor=expansion),
            MBTRBlock(dim=dims[0], depth=depth[0], channel=channels[5], patch_size=patch_size, mlp_dim=int(dims[0]*2))
        )

        self.layer4 = nn.Sequential(
            MV2Block(in_channels=channels[5], out_channels=channels[6], stride=2, expansion_factor=expansion),
            MBTRBlock(dim=dims[1], depth=depth[1], channel=channels[7], patch_size=patch_size, mlp_dim=int(dims[1]*4))
        )

        self.layer5 = nn.Sequential(
            MV2Block(in_channels=channels[7], out_channels=channels[8], stride=2, expansion_factor=expansion),
            MBTRBlock(dim=dims[2], depth=depth[2], channel=channels[9], patch_size=patch_size, mlp_dim=int(dims[2]*4))
        )

        self.last_conv = Conv1x1BNActivation(channels[9], channels[10])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=channels[10], out_features=num_classes)

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


def MBTR():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MBTR(dims, channels, num_classes=1000)
