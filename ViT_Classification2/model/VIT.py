import numpy as np
import torch
import torch.nn as nn
from image_helpers import patchify
from einops import repeat, rearrange

from model.VITBlock import MyViTBlock


class MyViT(nn.Module):
    def __init__(self, chw=(1, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=10, mask_ratio=None):
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

        # mask ratio

        self.mask_ratio = mask_ratio

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        b, _, _, _ = images.shape
        cls_tokens = repeat(self.class_token, 'n e -> b n e', b=b)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # positional embeddings
        pos_embed = repeat(self.pos_embed, 'n e -> b n e', b=b)
        out = tokens + pos_embed

        out = self.maskTokens(out)

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

    def maskTokens(self, tokens):
        if self.mask_ratio is None:
            return tokens

        # extract the class tokens
        cls_tokens = tokens[:, 0]

        # rearranging to add the first position in the patch dimention
        cls_tokens = repeat(cls_tokens, 'b e -> b p e', p=1)

        # get the patches
        patches = tokens[:, 1:]

        # shuffle the patches
        # np.random.shuffle(patches)

        order = np.array(range(patches.shape[1]))
        np.random.shuffle(order)

        # # in-place changing of values
        patches[:, np.array(range(patches.shape[1])), :] = patches[:, order, :]

        # only keep patches based on the amount of mask ration

        patches = patches[:, :int(
            np.floor(patches.shape[1] * (1-self.mask_ratio)))]

        # append the class tokens to the first position
        maskedTokens = torch.cat([cls_tokens, patches], dim=1)

        return maskedTokens
