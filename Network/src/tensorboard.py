from typing import Tuple, Optional, Literal
import torch
from .dataset import Dataset
from .network import Network
import numpy as np
import matplotlib.pyplot as plt

class TensorBoard:
    def __init__(self, dataset:Dataset, device:str, dirname:str, strtime:str, prefix:str='Net') -> None:
        self.dataset = dataset
        self.fig = f'{dirname}/{strtime}/{prefix}_{strtime}' + '_{epoch:03}.png'
        self.device = device


    def create(self, epoch:int, network:Network, unit:Optional[Literal['rad', 'm']]=None, save:bool=False) -> None:
        network.eval()
        stacks, targets = self.cuda(*next(self.dataset.iter()))
        
        with torch.no_grad():
            outputs, loss = network.call(stacks, targets)
        
        stacks, targets, outputs, loss = self.numpy(stacks, targets, outputs, loss)

        stacks, targets, outputs = self.dataset.unscale_input(stacks), self.dataset.unscale_target(targets), self.dataset.unscale_target(outputs)

        nrows, ncols = 3, stacks.shape[0]

        labels = ['stacks', 'targets', 'outputs']
        mask = {'3' : stacks == 0,
                '2' : np.max(stacks == 0, axis=0)} 
        
        datas = [stacks, targets, outputs]
        vmin, vmax = stacks.min(), stacks.max()
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))

        for axe, label, data in zip(axes, labels, datas):
            for idx, ax in enumerate(axe):
                if data.ndim == 3:
                    ifg = ax.imshow(np.ma.array(data[idx,], mask=mask[f'{data.ndim}'][idx,]), vmin=vmin, vmax=vmax, cmap='RdBu_r')
                
                elif data.ndim == 2:
                    if not idx:
                        ifg = ax.imshow(np.ma.array(data, mask=mask[f'{data.ndim}']), vmin=vmin, vmax=vmax, cmap='RdBu_r')
                    else:
                        ax.axis('off')
                
                if not idx:
                    ax.set_ylabel(label)
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle(f'epoch: {epoch:03d}, loss: {loss:0.5f}')
                     
        fig.subplots_adjust(right=0.925)
        axe = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(ifg, cax=axe, ticks=None)
        cbar.ax.set_title(f'[{unit}]' if unit else '')
    
        if save:
            fig.savefig(self.fig.format(epoch=epoch), dpi=750, bbox_inches='tight')
        plt.show()
    
    def cuda(self, *args:torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (arg.to(self.device) for arg in args)
    
    def numpy(self, *args:torch.Tensor) -> Tuple[np.ndarray, ...]:
        return (arg.detach().cpu().squeeze().numpy() for arg in args)