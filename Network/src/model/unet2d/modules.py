import torch
import torch.nn as nn

class ConvBock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(outdim),
            nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(outdim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class DeConvBlock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(indim, outdim, kernel_size=2, stride=2, ),
            nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(outdim),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class Concatenate(nn.Module):
    def __init__(self, dim:int=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y), dim=self.dim)


