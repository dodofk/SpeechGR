"""SpecAugment utilities shared by encoders."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking


class SpecAugmentLB(torch.nn.Module):
    """SpecAugment module (LibriSpeech style) used by Whisper encoder."""

    def __init__(self, time_warp_param=80, freq_mask_param=27, time_mask_param=100):
        super().__init__()
        self.time_warp_param = time_warp_param
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param=time_mask_param)

    def forward(self, spec):
        batch, n_mels, T = spec.shape
        if T <= 2 * self.time_warp_param:
            warped_spec = spec
        else:
            center = torch.randint(self.time_warp_param, T - self.time_warp_param, (batch,))
            warped = []
            for i, c in enumerate(center):
                left = spec[i, :, :c].unsqueeze(0)
                right = spec[i, :, c:].unsqueeze(0)
                left = F.pad(left, (self.time_warp_param, 0))[:, :, : c + self.time_warp_param]
                right = F.pad(right, (0, self.time_warp_param))[:, :, : T - c + self.time_warp_param]
                warped.append(
                    torch.cat(
                        [left[:, :, :c], right[:, :, self.time_warp_param :]], dim=2
                    )
                )
            warped_spec = torch.cat(warped, dim=0)

        warped_spec = self.freq_mask(warped_spec)
        warped_spec = self.time_mask(warped_spec)
        return warped_spec


__all__ = ["SpecAugmentLB"]
