from typing import Tuple,  Literal
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model import Unet2D, Unet3D

class Network(nn.Module):
    def __init__(self, backbone:Literal['unet2d', 'unet3d']='unet2d', dim:int=64, 
                 lr:float=2e-4, betas:Tuple[float, float]=(0.5, 0.999), distributed:bool=False, ngpus:int=1, mixed:bool=False) -> None:
        super().__init__()

        self.model = Unet2D(10, dim) if backbone == 'unet2d' else \
                     Unet3D(1, dim) if backbone == 'unet3d' else None
        
        if distributed:
            self.model = nn.DataParallel(self.model, list(range(ngpus)))

        self.optim = optim.Adam(params=self.model.parameters(),
                                       lr=lr,
                                       betas=betas)
        
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5, verbose=True)

        self.loss = nn.L1Loss()

        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed)

    def process(self, ifgs:torch.Tensor, targets:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.optim.zero_grad() # set_to_none=True here can modestly improve performance

        outputs, loss = self.call(ifgs, targets)

        self.backward(loss)

        return outputs, loss

    def call(self, ifgs:torch.Tensor, targets:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.scaler.is_enabled()):
            outputs = self(ifgs)
            loss = self.loss(outputs, targets)
        
        return outputs.to(torch.float32), loss

    def forward(self, ifgs:torch.Tensor) -> torch.Tensor:
        outputs = self.model(ifgs)
        return outputs

    def backward(self, loss:torch.Tensor) -> None:
        # loss.backward()
        # self.optim.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update() 

    def save(self, checkpointdir:str) -> None:
        filedir = f'{checkpointdir}/{self.getName()}.pth'
        print(f'Saving checkpoint to {filedir}')
        torch.save({
            'model' : self.model.state_dict(),},
            filedir)
        
    def load(self, checkpointdir:str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        filedir = f'{checkpointdir}/{self.getName()}.pth' 
        print(f'Loading checkpoint from {filedir}')
        if os.path.exists(filedir):
            data = torch.load(filedir, device)
            self.model.load_state_dict({key.replace('module.', '') : value for key, value in data['model'].items()})
        else:
            print(f'No saved model found under {filedir}.')

    def getName(self) -> str:
        if hasattr(self.model, 'module'):
            return self.model.module.__class__.__name__
        
        return self.model.__class__.__name__

    
    
    

