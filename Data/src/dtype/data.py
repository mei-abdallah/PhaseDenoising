from typing import Union, Literal, Optional, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
import bz2, pickle, os
from skimage.transform import resize
from scipy.stats import linregress
from ..objects import Sensor

plt.rc('font', **{'family' : 'serif', 
                  'weight' : 'normal'})

class Data(np.ndarray):
    isphase = True
    
    def __new__(cls, data:np.ndarray, dtype:np.dtype=np.float32) -> 'Data':
        return np.asarray(data, dtype=dtype).view(cls)
    
    def wrap(self, value:int=2) -> 'Data':
        """Wrap the values of a Data array around a circle with a given period of N PI"""
        return type(self)(np.mod(self.torad() + np.pi, value * np.pi) - (0.5 * value * np.pi))
    
    def tometer(self, wavelength:float=0.0562) -> 'Data':
        """Convert the phase in radians to displacment in meters using the wave length in meters"""
        if not self.isphase:
            return self
        
        self.isphase = False
        self *= (wavelength / (4 * np.pi))
        return self
    
    def torad(self, wavelength:float=0.0562) -> 'Data':
        """Convert the displacment to phase in radians using the wave length in meters"""
        if self.isphase:
            return self
        
        self.isphase = True
        self *= ((4 * np.pi) / wavelength)
        return self
    
    def square(self) -> 'Data':
        """Returns an array where each element is squared"""
        return type(self)(self ** 2)
    
    def diff(self, axis:int=-1) -> 'Data':
        """Calculate the difference between two successive elements along an axis"""
        return type(self)(np.diff(self, n=1, axis=axis))
    
    def relative(self, mode:Literal['ifgs', 'timeseries']='ifgs') -> 'Data':
        """Calculate the relative difference between two successive elements along an axis"""
        if mode == 'ifgs':
            return self.diff()
        return self
    
    def cumsum(self, axis:Optional[Union[int, Tuple[int]]]=None) -> 'Data':
        """Return the cumulative sum of all previous elements along an axis"""
        self.data = np.cumsum(self.data, axis=axis)
        return self
    
    def tile(self, axis:Union[int, Sequence[int]]) -> 'Data':
        """Tile the input array by repeating it across multiple dimensions."""
        return type(self)(np.tile(self, axis))
    
    def repeat(self, repeats:int, axis:int) -> 'Data':
        """Repeat the values of a Data array along one or more axes"""
        return type(self)(np.repeat(self, repeats, axis))
    
    def unsqueeze(self, axis:int=-1) -> 'Data':
        """Remove dimensions of size one from the shape of an array."""
        return type(self)(np.expand_dims(self, axis))

    def absolute(self) -> 'Data':
        """Return the element-wise absolute value of the input array."""
        return type(self)(np.abs(self))

    def resize(self, shape:Tuple[int, int]) -> 'Data':
        """Resize the image to new width and height"""
        return type(self)(resize(self, shape))
    
    def nanmin(self, axis:Optional[int]=None) -> 'Data':
        return type(self)(np.nanmin(self, axis))
    
    def nanmax(self, axis:Optional[int]=None) -> 'Data':
        return type(self)(np.nanmax(self, axis))
    
    def crop(self, shape:Tuple[int, int], xstart:Optional[int]=None, ystart:Optional[int]=None)-> 'Data':
        """Crops the image to new width and height at specified coordinates"""
        if shape > self.shape:
            raise ValueError("Shape must be smaller than or equal to current size")
        if shape == self.shape:
            return self
        
        xstart = xstart or np.random.randint(0, self.shape[1] - shape[1])
        ystart = ystart or  np.random.randint(0, self.shape[0] - shape[0])
        return type(self)(self[ystart:ystart + shape[0], 
                               xstart:xstart + shape[1]])

    def save(self, filename:str) -> None:
        """Save the Data object to a file with .pkl extension"""
        with bz2.BZ2File(filename, 'wb') as file:
            pickle.dump(self, file)
        file.close()

    def plot(self, 
             title:Optional[str]=None, 
             cmap:Literal['terrain', 'jet', 'gray', 'coolwarm', 'seisemic', 'bwr', 'RdBu_r']='RdBu_r', 
             unit:Optional[Literal['rad', 'm']]=None, 
             save:bool=False, 
             vmin:Optional[float]=None, 
             vmax:Optional[float]=None, 
             show:bool=True) -> None:
        """ plot the data"""
        plt.rc('font', **{'size'   : 10})
        data = self.squeeze()
        if data.ndim == 1:
            return

        if self.dtype == complex:
            data = np.angle(data)

        if unit is not None and cmap != 'terrain':
            unit = 'rad' if self.isphase else 'm'

        vmin, vmax = vmin or data.min(), vmax or data.max()
        
        # if unit == 'rad' and np.max(np.abs(self)) >= np.pi:
        #     factor = np.ceil(np.max(np.abs(self)) / np.pi)
        #     vmin, vmax = -factor * np.pi, factor * np.pi

        if data.ndim == 2:
            fig, axe = plt.subplots(1, 1, figsize=(5, 5))
            img = axe.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axe.set_xticks([0, data.shape[1]])
            axe.set_yticks([0, data.shape[0]])
            axe.set_title(title if title else data.__class__.__name__)

        elif data.ndim == 3:
            nslcs = data.shape[-1]
            ncols = int(np.ceil(np.sqrt(nslcs)))
            nrows = int(np.ceil(nslcs/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 1, 2.5 * nrows + 1))
            axes = np.array([[axes]]) if nrows == ncols == 1 else axes[np.newaxis, ] if nrows == 1 else axes[..., np.newaxis] if ncols == 1 else axes

            for nrow in range(nrows):
                for ncol in range(ncols):
                    if ncol + nrow * ncols < nslcs:
                        img = axes[nrow, ncol].imshow(data[..., ncol + nrow * ncols], vmin=vmin, vmax=vmax, cmap=cmap)
                    else:
                        axes[nrow, ncol].axis('off')
                    
                    axes[nrow, ncol].set_xticks([0, data.shape[1]] if nrow == nrows - 1 else [])
                    axes[nrow, ncol].set_yticks([0, data.shape[0]] if ncol == 0 else [])
                    
            fig.suptitle(title if title else self.__class__.__name__, y=1.01)

        fig.tight_layout()
        
        fig.subplots_adjust(right=0.9)
        axe = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(img, cax=axe, ticks=None)
        cbar.ax.set_title(f'[{unit}]' if unit else '')
        

        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/plot_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    def draw(self, 
             points:Optional[Sequence[Tuple[int, int]]]=None, 
             title:Optional[str]=None, 
             unit:Optional[Literal['rad', 'm']]='rad', 
             save:bool=False, 
             vmin:Optional[float]=None, 
             vmax:Optional[float]=None, 
             show:bool=True)->None:
        """Draw at specfic level the image."""
        plt.rc('font', **{'size'   : 10})
        data = self.squeeze()
        
        if data.ndim != 3:
            return
        
        if points is None:
            points = [np.unravel_index(data.argmax(), data.shape)[:-1]]
            
        if self.dtype == complex:
            data = np.angle(data)

        
        if unit is not None:
            unit = 'rad' if self.isphase else 'm'

        vmin, vmax = vmin or data.min(), vmax or data.max()
        # print(vmin, vmax)
        nrows, ncols = len(points), 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 2.5 * nrows), sharex=True)
        axes = np.ravel(axes)

        for idx, axe in enumerate(axes):
            values = data[points[idx]]
            axe.plot(values, color='darkorange', marker='o', linestyle='dashed')
            axe.set_ylim(vmin * (1.15 if vmin <= 0.0 else 0.85), vmax * (1.15 if vmax >= 0.0 else 0.85))
            axe.set_ylabel(f'[{unit}]' if unit else '')
            axe.text(0.5, 0.9, f'y = {((values[-1] - values[0]) / len(values)):0.3f} x + {values[0]:0.3f}', transform=axe.transAxes,  # 'y = {a:f} x + {b:f}'.format(a=((values[-1] - values[0]) / len(values)), b=values[0])
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                              ))
            axe.text(0.85, 0.9, 'Point = ({}, {})'.format(*points[idx]), transform=axe.transAxes,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                              ))
        
        
        fig.suptitle(title if title else self.__class__.__name__, y=1.01)
        fig.tight_layout()

        # Save figure
        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/draw_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    def hist(self, bins:int=10, unit:Optional[Literal['rad', 'm']]='rad', save:bool=False, show:bool=False) -> None:
        """Plot the histogram of the field."""
        plt.rc('font', **{'size'   : 10})
        data = self.squeeze().flatten()

        to = 1.0

        if self.isphase:
            to = (4 * np.pi) / 0.0562
        
        if unit is not None:
            unit = 'rad' if self.isphase else 'm'

        fig = plt.figure(figsize=(7.5, 5.0))
        axe = fig.add_subplot()
        axe.grid(True)
        axe.hist(data, bins, color='darkorange')
        axe.set_yscale("log")
        axe.set_xlabel(f'[{unit}]' if unit else '')
        axe.set_xlim(-0.05 * to, 0.05 * to)
        axe.set_ylabel("Frequency")

        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/hist_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

class MaskedData(np.ma.MaskedArray):
    isphase = True
    def __new__(cls, data:Union[np.ndarray, np.ma.MaskedArray], mask:Optional[np.ndarray]=None, dtype:np.dtype=np.float32) -> 'MaskedData':
        # Create a new instance of MaskedRaster from the input array
        return super().__new__(cls, data, mask, dtype=dtype)
    
    def wrap(self, value:int=2) -> 'MaskedData':
        """Wrap the values of a Data array around a circle with a given period of N PI"""
        return type(self)(np.ma.mod(self + np.pi, value * np.pi) - (0.5 * value * np.pi))
    
    def tometer(self, wavelength:float=0.055) -> 'Data':
        """Convert the phase in radians to displacment in meters using the wave length in meters"""
        if not self.isphase:
            return self
        
        self.isphase = False
        self *= (wavelength / (4 * np.pi))
        return self
    
    def torad(self, wavelength:float=0.055) -> 'Data':
        """Convert the displacment to phase in radians using the wave length in meters"""
        if self.isphase:
            return self
        
        self.isphase = True
        self *= ((4 * np.pi) / wavelength)
        return self
    
    def square(self) -> 'MaskedData':
        """Returns an array where each element is squared"""
        return type(self)(self ** 2)
    
    def diff(self, axis:int=-1) -> 'MaskedData':
        """Calculate the difference between two successive elements along an axis"""
        return type(self)(np.ma.diff(self, n=1, axis=axis))
    
    def relative(self, mode:Literal['ifgs', 'timeseries']='ifgs') -> 'Data':
        """Calculate the relative difference between two successive elements along an axis"""
        if mode == 'ifgs':
            return self.diff()
        return self
    
    def cumsum(self, axis:Optional[Union[int, Tuple[int]]]=None) -> 'Data':
        """Return the cumulative sum of all previous elements along an axis"""
        self.data = np.cumsum(self.data, axis=axis)
        return self
    
    def tile(self, axis:Union[int, Sequence[int]]) -> 'Data':
        """Tile the input array by repeating it across multiple dimensions."""
        return type(self)(np.tile(self, axis))
    
    def repeat(self, repeats:int, axis:int) -> 'Data':
        """Repeat the values of a Data array along one or more axes"""
        return type(self)(np.repeat(self, repeats, axis))
        
    def unsqueeze(self, axis:int=-1) -> 'MaskedData':
        """Remove one dimension at specified position and return as numpy array"""
        return type(self)(np.expand_dims(self, axis))
    
    def absolute(self) -> 'MaskedData':
        """Return the absolute value of all elements in the array"""
        return type(self)(np.abs(self))
    
    def resize(self, shape:Tuple[int, int]) -> 'MaskedData':
        """Resize the array by changing its dimensions"""
        data = resize(self.data, shape)
        mask = resize(self.mask, shape)
        return type(self)(data, mask)
    
    def nanmin(self, axis:Optional[int]=None) -> 'MaskedData':
        return type(self)(np.nanmin(self, axis))
    
    def nanmax(self, axis:Optional[int]=None) -> 'MaskedData':
        return type(self)(np.nanmax(self, axis))
    
    def crop(self, shape:Tuple[int, int])-> 'MaskedData':
        """Crops the image to the desired size"""
        if shape > self.shape:
            raise ValueError("Shape must be smaller than or equal to current size")
        if shape == self.shape:
            return self
        
        xstart = np.random.randint(0, self.shape[1] - shape[1])
        ystart = np.random.randint(0, self.shape[0] - shape[0])
        return type(self)(self[ystart:ystart + shape[0], 
                               xstart:xstart + shape[1]])

    def save(self, filename:str) -> None:
        """Save the Data object to a file with .pkl extension"""
        with bz2.BZ2File(filename, 'wb') as file:
            pickle.dump(self, file)
        file.close()
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} (\n{super().__repr__()})>"
    

    def plot(self, 
             title:Optional[str]=None, 
             cmap:Literal['terrain', 'jet', 'gray', 'coolwarm', 'seisemic', 'bwr', 'RdBu_r']='RdBu_r', 
             unit:Optional[Literal['rad', 'm']]=None, 
             save:bool=False, 
             vmin:Optional[float]=None, 
             vmax:Optional[float]=None,
             show:bool=True) -> None:
        """ plot the data"""
        plt.rc('font', **{'size'   : 10})

        data = self.squeeze()
        if data.ndim == 1:
            return

        if self.dtype == complex:
            data = np.angle(data)

        if unit is not None and cmap != 'terrain':
            unit = 'rad' if self.isphase else 'm'

        vmin, vmax = vmin or data.min(), vmax or data.max()
        
        # if unit == 'rad' and np.max(np.abs(self)) >= np.pi:
        #     factor = np.ceil(np.max(np.abs(self)) / np.pi)
        #     vmin, vmax = -factor * np.pi, factor * np.pi

        if data.ndim == 2:
            fig, axe = plt.subplots(1, 1, figsize=(5, 5))
            img = axe.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axe.set_xticks([0, data.shape[1]])
            axe.set_yticks([0, data.shape[0]])
            axe.set_title(title if title else data.__class__.__name__)

        elif data.ndim == 3:
            nslcs = data.shape[-1]
            ncols = int(np.ceil(np.sqrt(nslcs)))
            nrows = int(np.ceil(nslcs/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 1, 2.5 * nrows + 1))
            axes = np.array([[axes]]) if nrows == ncols == 1 else axes[np.newaxis, ] if nrows == 1 else axes[..., np.newaxis] if ncols == 1 else axes

            for nrow in range(nrows):
                for ncol in range(ncols):
                    if ncol + nrow * ncols < nslcs:
                        img = axes[nrow, ncol].imshow(data[..., ncol + nrow * ncols], vmin=vmin, vmax=vmax, cmap=cmap)
                    else:
                        axes[nrow, ncol].axis('off')
                    
                    axes[nrow, ncol].set_xticks([0, data.shape[1]] if nrow == nrows - 1 else [])
                    axes[nrow, ncol].set_yticks([0, data.shape[0]] if ncol == 0 else [])
                    
            fig.suptitle(title if title else self.__class__.__name__, y=1.01)

        fig.tight_layout()
        
        fig.subplots_adjust(right=0.9)
        axe = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(img, cax=axe, ticks=None)
        cbar.ax.set_title(f'[{unit}]' if unit else '')
        

        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/plot_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    def draw(self, 
             points:Optional[Sequence[Tuple[int, int]]]=None, 
             title:Optional[str]=None, 
             unit:Optional[Literal['rad', 'm']]='rad', 
             save:bool=False, 
             vmin:Optional[float]=None, 
             vmax:Optional[float]=None,
             show:bool=True)->None:
        """Draw at specfic level the image."""
        plt.rc('font', **{'size'   : 10})
        data = self.squeeze()
        
        if data.ndim != 3:
            return
        
        if points is None:
            points = [np.unravel_index(data.argmax(), data.shape)[:-1]]
            
        if self.dtype == complex:
            data = np.angle(data)

        
        if unit is not None:
            unit = 'rad' if self.isphase else 'm'

        vmin, vmax = vmin or data.min(), vmax or data.max()

        nrows, ncols = len(points), 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 2.5 * nrows), sharex=True)
        axes = np.ravel(axes)

        for idx, axe in enumerate(axes):
            values = data[points[idx]]
            axe.plot(values, color='darkorange', marker='o', linestyle='dashed')
            axe.set_ylim(vmin * (1.15 if vmin <= 0.0 else 0.85), vmax * (1.15 if vmax >= 0.0 else 0.85))
            axe.set_ylabel(f'[{unit}]' if unit else '')
            axe.text(0.5, 0.9, f'y = {((values[-1] - values[0]) / len(values)):0.3f} x + {values[0]:0.3f}', transform=axe.transAxes,  # 'y = {a:f} x + {b:f}'.format(a=((values[-1] - values[0]) / len(values)), b=values[0])
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                              ))
            axe.text(0.85, 0.9, 'Point = ({}, {})'.format(*points[idx]), transform=axe.transAxes,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                              ))
        
        
        fig.suptitle(title if title else self.__class__.__name__, y=1.01)
        fig.tight_layout()

        # Save figure
        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/draw_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    def hist(self, 
             bins:int=10, 
             unit:Optional[Literal['rad', 'm']]='rad', 
             save:bool=False,
             show:bool=True) -> None:
        """Plot the histogram of the field."""
        plt.rc('font', **{'size'   : 10})
        data = self.squeeze().flatten()

        to = 1.0

        if self.isphase:
            to = (4 * np.pi) / 0.0562
        
        if unit is not None:
            unit = 'rad' if self.isphase else 'm'

        fig = plt.figure(figsize=(7.5, 5.0))
        axe = fig.add_subplot()
        axe.grid(True)
        axe.hist(data, bins, color='darkorange')
        axe.set_yscale("log")
        axe.set_xlabel(f'[{unit}]' if unit else '')
        axe.set_xlim(-0.05 * to, 0.05 * to)
        axe.set_ylabel("Frequency")

        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/hist_{data.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        
        if show:
            plt.show()

    


