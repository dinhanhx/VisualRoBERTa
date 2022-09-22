from typing import Tuple
import torch.nn as nn

from einops.layers.torch import Rearrange
from src.resnet import ResNetMSA


class PatchEmbedding(nn.Module):
    """ PatchEmbedding like ViT 
        with ResNetMSA for learning patches via CNN
    """
    def __init__(self, image_size: Tuple = (256, 256),
                 patch_size: Tuple = (32, 32),
                 emb_dim = 768):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        self.num_patches = (image_height // patch_height) * \
            (image_width // patch_width)

        self.embedding = nn.Sequential(
            Rearrange('B C (NPH PH) (NPW PW) -> (B NPH NPW) C PH PW',
                      PH=patch_height, PW=patch_width),
            ResNetMSA()
        )

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024, emb_dim),
            Rearrange('(B NP) E -> B NP E', NP=self.num_patches),
            nn.LayerNorm(emb_dim, eps=1e-12)
        )

    def forward(self, x):
        """ x shape [B, C, H, W]
            x out shape [B, NP, E]
        """
        x = self.embedding(x)
        x = self.output_proj(x)
        return x


class RegionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
