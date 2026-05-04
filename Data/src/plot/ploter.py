from typing import Literal, Sequence, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from ..base import Stack
from ..dtype import Data

plt.rc('font', **{'family' : 'serif', 
                  'weight' : 'normal'})

class Ploter(object):
    def __init__(self, stack:Stack) -> None:
        self.stack = stack
    
    def run(self, 
            labels:Sequence[Literal['topo', 'turb', 'tropo', 'orbit', 'trend', 'noisy']], 
            unit:Literal['rad', 'm', 'cm'], 
            vmin:Optional[float]=None, 
            vmax:Optional[float]=None, 
            save:bool=False, 
            show:bool=True,
            cmap:Literal['coolwarm', 'seisemic', 'bwr', 'RdBu_r']='RdBu_r') -> None:
        """Plot the data in the stack and optionally save it or display it"""
        plt.rc('font', **{'size'   : 20})

        alpha = 'abcdefghijklmnopqrstuvwxyz'

        to = {'m' : 'tometer', 'cm' : 'tometer', 'rad' : 'torad'}
        getters = [f'get{label.title()}' for label in labels]
        data = [getattr(getattr(self.stack, getter)(), to[unit])() * (1.0 if unit != 'cm' else 100) for getter in getters]
        
        length = data[0].shape[-1]
        nrows, ncols = len(data),  length + 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.65, nrows * 3.65), gridspec_kw={'width_ratios': [1] * length +  [2.0, 0.1]})

        if nrows == 1:
            axes = axes[np.newaxis, ...]
  
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        if vmin is not None:
            vmin = getattr(Data(vmin), to[unit])() * (1.0 if unit != 'cm' else 100)
            
        if vmax is not None:
            vmax = getattr(Data(vmax), to[unit])() * (1.0 if unit != 'cm' else 100)

        for nrow in range(nrows):
            for ncol in range(ncols):
                if not ncol:
                    axes[nrow, ncol].set_ylabel(f'({alpha[nrow]})') 

                if ncol < ncols - 2:
                    img = self.imshow(axes[nrow, ncol], data[nrow][..., ncol], cmap, vmin * (0.5 if labels[nrow] != 'noisy' else 1.0), vmax * (0.5 if labels[nrow] != 'noisy' else 1.0))

            # Add histogram  
            self.hist(axes[nrow, -2], data[nrow], unit, vmin, vmax, bool(nrow == nrows - 1))

            # Add colorbar
            self.colorbar(self.move(axes[nrow, -1], x=0.02), img, unit)

        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/plot_{self.stack.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    def imshow(self, axe:Axes, data:np.ndarray, cmap:str='RdBu_r', vmin:Optional[float]=None, vmax:Optional[float]=None)-> AxesImage:
        """Add an image to a specific axis with given color scale"""
        img = axe.imshow(data, cmap, vmin=vmin, vmax=vmax)
        axe.set_xticks([]); axe.set_yticks([]) # remove ticks
        return img
    
    def hist(self, axe:Axes, data:np.ndarray, unit:str, vmin:Optional[float]=None, vmax:Optional[float]=None, addticks:bool=True, bins:int=10, color:str='silver') -> None:
        """Add a histogram of the data on the specified axis"""
        axe.grid(True)
        axe.hist(data.flatten(), bins, histtype='bar', edgecolor='black', color=color, rwidth=0.8)
        axe.yaxis.tick_right()
        axe.set_yscale("log")
        axe.set_xlim(vmin, vmax)
        axe.set_xlabel(f'[{unit}]')
        if not addticks:
            axe.set_xlabel('')
            axe.set_xticklabels('')
            

    def colorbar(self, axe:Axes, img:AxesImage, unit:str)-> None:
        """Create and add a color bar to the figure"""
        fig = img.get_figure()
        cbar = fig.colorbar(img, cax=axe, label=f'[{unit}]', aspect=25, pad=0., shrink=0.8)
        # cbar.solids.set_edgecolor("face")
        # cbar.ax.set_title(f'[{unit}]')

    def move(self, axe:Axes, x:float=0.0, y:float=0.0)-> Axes:
        box = axe.get_position(True)
        box.x0 = box.x0 + x; box.y0 = box.y0 + y
        box.x1 = box.x1 + x; box.y1 = box.y1 + y
        axe.set_position(box)
        return axe