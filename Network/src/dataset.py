from typing import Literal, Tuple, Iterator
import torch
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import configparser
from Data.src import TimeSeries

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir:str, source:Literal['mogi']='mogi', mode:Literal['ts', 'ifg']= 'ts', phase:Literal['train', 'valid']='train', 
                 ismasked:bool=False, agument:bool=False) -> None:
        super().__init__()
        self.files = sorted(glob(f'{datadir}/{source}/{phase}/*.pkl'))
        self.mode = mode
        self.ismasked = ismasked
        self.augment = agument
        self.config = configparser.ConfigParser()
        self.config.read(f'{datadir}/{source}/config.ini')
        
    def __len__(self) -> int:
        # return len(self.files)
        return 10
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = self.files[index]

        stack = TimeSeries.read(filepath)
        inputs = self.get_input(stack)
        
        targets = self.get_target(stack)

        if self.augment and np.random.rand() > 0.5:
            mode = np.random.choice(['no', 'up-down', 'left-right', 'both'])
            inputs, targets = self.flip(inputs, mode), self.flip(targets, mode)

        return self.to_tensor(inputs), self.to_tensor(targets)
    
    def get_input(self, stack:TimeSeries) -> np.ndarray:
        if self.mode == 'ts':
            inputs = stack.getNoisy().transpose((2, 0, 1)).tometer() 
        elif self.mode == 'ifg':
            inputs = stack.getIfgs().getNoisy().transpose((2, 0, 1)).tometer()
        else:
            raise ValueError(f"Invalid mode {self.mode}")
        inputs = self.scale_input(inputs)
        return inputs
    
    def get_target(self, stack:TimeSeries) -> np.ndarray:
        if self.mode == 'ts':
            targets = stack.getTrend().transpose((2, 0, 1)).tometer() 
        elif self.mode == 'ifg':
            targets = stack.getIfgs().getTrend().transpose((2, 0, 1)).tometer()
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        targets = self.scale_target(targets)
        return targets
    
    def get_dem(self, stack:TimeSeries) -> np.ndarray:
        dem = stack.getDemHeights().unsqueeze(0)
        dem = self.scale_dem(dem)
        return dem
    
    def get_input_dem(self, stack:TimeSeries) -> np.ndarray:
        inputs = self.get_input(stack)
        dem = self.get_dem(stack)
        return np.concatenate([inputs, dem], axis=0)

    def flip(self, data:np.ndarray, mode:Literal['no', 'up-down', 'left-right', 'both']='no') -> np.ndarray:
        if mode == 'no':
            return data
        
        elif mode == 'up-down':
            # vertical flipping (axis=-2)
            return data[..., ::-1, :]
        
        elif mode == 'left-right':
            # horizontal flipping (axis=-1)
            return data[..., :, ::-1]
        
        else:
            # both directions flipped simultaneously along the two axes (-2 and -1 respectively).
            return data[..., ::-1, ::-1]

    
    def scale_input(self, data:np.ndarray) -> np.ndarray:
        data = self.scale(data)
        return data
    
    def scale_target(self, data:np.ndarray) -> np.ndarray:
        data = self.normlize(data)
        return data
    
    def scale_dem(self, data:np.ndarray) -> np.ndarray:
        data = self.standardize(data)
        return data
    
    def unscale_input(self, data:np.ndarray) -> np.ndarray:
        data = self.unscale(data)
        return data
    
    def unscale_target(self, data:np.ndarray) -> np.ndarray:
        data = self.denormlize(data)
        return data

    def scale(self, data:np.ndarray) -> np.ndarray:
        """Scale data array between 0 and 1"""
        data = data.copy()
        data -= float(self.config[self.mode]['min'])
        data *= (2.0 / (float(self.config[self.mode]['max']) - float(self.config[self.mode]['min'])))
        data -= 1.0
        return data
    
    def standardize(self, data:np.ndarray) -> np.ndarray:
        """Standardize data array between -1 and 1"""
        data = data.copy()
        data -= data.min()
        data *= (2 / data.max())
        data -= 1.0
        return data

    def normlize(self, data:np.ndarray) -> np.ndarray:
        """Normalize data array to have zero mean and unit variance"""
        data = data.copy()
        return data
    
    def unscale(self, data:np.ndarray) -> np.ndarray:
        """UnScale data array between min and max"""
        data = data.copy()
        data += 1.0
        data *= ((float(self.config[self.mode]['max']) - float(self.config[self.mode]['min'])) / 2.0)
        data += float(self.config[self.mode]['min'])
        return data
    
    def denormlize(self, data:np.ndarray) -> np.ndarray:
        """DeNormalize data array between min and max"""
        data = data.copy()
        return data
    
    def to_tensor(self, data:np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).contiguous().float()
    

    def iter(self, batch_size:int=1) -> Iterator[Tuple[torch.Tensor, ...]]:
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                num_workers=4,
                drop_last=True
            )

            for item in sample_loader:
                yield item