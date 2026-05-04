import torch
import torch.nn as nn
from .layers import EncoderBlock, DecoderBlock, NeckBlock, InOutBlock

class Unet2D(nn.Module):
    def __init__(self, channels:int, filters:int=32) -> None:
        super().__init__()
        self.input = InOutBlock(channels, filters)
        self.encoder1 = EncoderBlock(filters, filters)
        self.encoder2 = EncoderBlock(filters, filters*2)
        self.encoder3 = EncoderBlock(filters*2, filters*4)

        self.neck = NeckBlock(filters*4, filters*8)

        self.decoder1 = DecoderBlock(filters*8, filters*4)
        self.decoder2 = DecoderBlock(filters*4, filters*2)
        self.decoder3 = DecoderBlock(filters*2, filters)

        self.output = InOutBlock(filters, channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        e1, s1 = self.encoder1(x)
        e2, s2 = self.encoder2(e1)
        e3, s3 = self.encoder3(e2)

        neck = self.neck(e3)

        d1 = self.decoder1(neck, s3)
        d2 = self.decoder2(d1, s2)
        d3 = self.decoder3(d2, s1)

        out = self.output(d3)
        return out


