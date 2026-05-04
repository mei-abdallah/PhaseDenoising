from typing import Tuple
import torch
import torch.nn as nn
from .modules import ConvBock, DeConvBlock, Concatenate

class EncoderBlock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.conv = ConvBock(indim, outdim)
        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip
    
class NeckBlock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.neck = nn.Sequential(
            ConvBock(indim, outdim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.neck(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.deconv = DeConvBlock(indim, indim//2)
        self.cat = Concatenate()
        self.conv = ConvBock(indim, outdim)

    def forward(self, x:torch.Tensor, res:torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        # upsample the input feature map and concatenate it with decoded output
        # print(f'x: {x.shape}, res: {res.shape}')
        x = self.cat(x, res)
        return self.conv(x)

class InOutBlock(nn.Module):
    def __init__(self, indim:int, outdim:int) -> None:
        super().__init__()
        self.head = nn.Conv3d(indim, outdim, kernel_size=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.head(x)
    