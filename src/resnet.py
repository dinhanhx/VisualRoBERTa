import torch.nn as nn

from torch import einsum
from einops import rearrange


class Attention2d(nn.Module):
    """ MSA from https://github.com/xxxnell/how-do-vits-work
    """

    def __init__(self,
                 dim_in,
                 dim_out=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.0):
        """
        dim_in: number of channels in the input image
        dim_out: number of channels produced by the last convolution layer
        heads: parallel attention heads
        dim_head: number of channels produce by each head
        dropout: a dropout layer on the output of the last convolution layer
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        """
        Input:
            x (B, dim_in, H, W)
            mask ()
        Output:
            out (B, dim_out, H, W)
            attn (B, heads, H*W, H*W)
        """
        b, n, _, y = x.shape

        # Go throught conv
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d',
                                          h=self.heads),
                      qkv)
        
        # Scale dot product attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        
        # Go throught conv
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)
        out = self.to_out(out)
        return out


class ResNetBlock(nn.Module):
    """ ResNet from scratch
    https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#ResNet
    """
    def __init__(self, c_in, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block
                and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant
                if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(
                c_in, c_out, kernel_size=3, padding=1,
                stride=1 if not subsample else 2, bias=False
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value,
        # and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) \
            if subsample else None
        self.act_fn = nn.ReLU()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class ResNetMSA(nn.Module):
    """ A customized ResNet
    with MSA from https://github.com/xxxnell/how-do-vits-work
    as the last layer of the last block
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.g1 = nn.Sequential(
            ResNetBlock(16),
            ResNetBlock(16),
            ResNetBlock(16)
        )
        
        self.g2 = nn.Sequential(
            ResNetBlock(16, subsample=True, c_out=32),
            ResNetBlock(32),
            ResNetBlock(32)
        )
        
        self.g3 = nn.Sequential(
            ResNetBlock(32, subsample=True, c_out=64),
            ResNetBlock(64),
            Attention2d(64, heads=12)
        )

    def forward(self, x):
        """ x shape [B, C, H, W]
        """
        x = self.input_proj(x)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)
