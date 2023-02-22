import numpy as np
import torch
import torch.nn as nn
from image_helpers import patchify
from einops import repeat

from model.VITBlock import MyViTBlock


class MyViT(nn.Module):
    def __init__(self, chw=(1, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        self.patch_size = (chw[1] / self.n_patches, chw[2] / self.n_patches)

        # linear mapping
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # learnable class token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # positional embedding

        self.register_buffer('pos_embed', self.get_positional_embeddings(
            n_patches ** 2 + 1, hidden_d), persistent=False)

        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        b, _, _, _ = images.shape
        cls_tokens = repeat(self.class_token, 'n e -> b n e', b=b)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # positional embeddings
        pos_embed = repeat(self.pos_embed, 'n e -> b n e', b=b)
        out = tokens + pos_embed

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]
        out = self.mlp(out)
        return out

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(
                    i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
