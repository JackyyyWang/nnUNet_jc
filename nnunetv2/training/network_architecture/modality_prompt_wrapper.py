import torch
from torch import nn


class ModalityPromptWrapper(nn.Module):
    """Wraps a base segmentation network with per-modality prompt encoders."""

    def __init__(self, base_network: nn.Module, num_modalities: int):
        super().__init__()
        self.base_network = base_network
        self.prompt_encoders = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU(inplace=True)) for _ in range(num_modalities)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompts = []
        for m in range(x.shape[1]):
            prompts.append(self.prompt_encoders[m](x[:, m : m + 1]))
        x = torch.cat(prompts, dim=1)
        return self.base_network(x)

