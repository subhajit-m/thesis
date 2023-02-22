from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
import torch


def patchify(images, num_patches):
    # patches = rearrange(images, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
    patches = rearrange(
        images, 'b c (h s1) (w s2) -> b (s1 s2) (h w c)', s1=num_patches, s2=num_patches)
    return patches


# def patchify(images, n_patches):
#     n, c, h, w = images.shape

#     assert h == w, "Patchify method is implemented for square images only"

#     patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#     patch_size = h // n_patches

#     for idx, image in enumerate(images):
#         for i in range(n_patches):
#             for j in range(n_patches):
#                 patch = image[:, i * patch_size: (i + 1) * patch_size,
#                               j * patch_size: (j + 1) * patch_size]
#                 patches[idx, i * n_patches + j] = patch.flatten()
#     return patches
