"""Neural Cellular Automata model with optional conditioning.

Extends the base NCA architecture (Mordvintsev et al., 2020) with:
- Optional class-conditional embedding concatenated to perception
- reset_hidden() for zeroing hidden channels while preserving RGBA

When cond_dim=0 (default), behaves identically to the original model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCAModel(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128, fire_rate=0.5,
                 cond_dim=0, num_classes=0):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_n = hidden_n
        self.fire_rate = fire_rate
        self.cond_dim = cond_dim
        self.num_classes = num_classes

        # Fixed perception kernels (not learned)
        identify = np.outer([0, 1, 0], [0, 1, 0]).astype(np.float32)
        sobel_x = np.outer([1, 2, 1], [-1, 0, 1]).astype(np.float32) / 8.0
        sobel_y = sobel_x.T.copy()

        # Depthwise conv kernel: for each of the C input channels, apply 3 filters
        # Shape: [3*C, 1, 3, 3] for use with groups=C
        kernel = np.stack([identify, sobel_x, sobel_y])  # [3, 3, 3]
        kernel = np.tile(kernel, (channel_n, 1, 1))  # [3*C, 3, 3]
        kernel = kernel[:, np.newaxis, :, :]  # [3*C, 1, 3, 3]
        self.register_buffer("perception_kernel", torch.from_numpy(kernel))

        # Conditional embedding
        if cond_dim > 0 and num_classes > 0:
            self.cond_embedding = nn.Embedding(num_classes, cond_dim)

        # Update network: two 1x1 convolutions
        fc1_in = channel_n * 3 + cond_dim
        self.fc1 = nn.Conv2d(fc1_in, hidden_n, 1)
        self.fc2 = nn.Conv2d(hidden_n, channel_n, 1, bias=False)

        # Zero-initialize final layer so model starts as identity (critical)
        with torch.no_grad():
            self.fc2.weight.zero_()

    def perceive(self, x):
        """Apply fixed perception filters. Input [B,C,H,W] -> Output [B,3C,H,W]."""
        return F.conv2d(x, self.perception_kernel, padding=1, groups=self.channel_n)

    @staticmethod
    def alive_mask(x):
        """A cell is alive if any cell in its 3x3 neighborhood has alpha > 0.1."""
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1

    @staticmethod
    def reset_hidden(x):
        """Zero hidden channels (4:) while preserving RGBA (0:4)."""
        x = x.clone()
        x[:, 4:] = 0.0
        return x

    def forward(self, x, cond_ids=None, fire_rate=None, step_size=1.0):
        """Single NCA update step with stochastic mask and alive masking."""
        if fire_rate is None:
            fire_rate = self.fire_rate

        pre_mask = self.alive_mask(x)

        y = self.perceive(x)  # [B, 3C, H, W]

        # Concatenate conditioning
        if self.cond_dim > 0:
            B, _, H, W = y.shape
            if cond_ids is not None:
                emb = self.cond_embedding(cond_ids)  # [B, cond_dim]
                emb = emb[:, :, None, None].expand(B, self.cond_dim, H, W)
            else:
                emb = y.new_zeros(B, self.cond_dim, H, W)
            y = torch.cat([y, emb], dim=1)  # [B, 3C + cond_dim, H, W]

        dx = self.fc2(F.relu(self.fc1(y))) * step_size

        # Stochastic update: same mask across all channels for a given cell
        update_mask = (torch.rand_like(x[:, :1, :, :]) <= fire_rate).float()
        x = x + dx * update_mask

        # Alive mask: AND of pre-update and post-update masks
        post_mask = self.alive_mask(x)
        life_mask = (pre_mask & post_mask).float()

        return x * life_mask
