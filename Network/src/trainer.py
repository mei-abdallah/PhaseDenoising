from typing import Tuple, Literal, Optional
from torch.utils.data import DataLoader
import torch, time, random, numpy as np
from .dataset import Dataset
from .network import Network
from .metrics import Metric
from .checkpoint import CheckPoint
from .logger import CSVLogger, Hyperparameters
from .callbacks import Callback
from .progbar import ProgBar
from .tensorboard import TensorBoard

torch.cuda.empty_cache()

class Trainer:
    def __init__(self, datadir:str, source:Literal['mogi']='mogi', backbone:Literal['unet2d', 'unet3d']='unet2d', dim:int=64,
                 mode:Literal['ts', 'ifg']= 'ts', ismasked:bool=False, augment:bool=False, batch_size:int=32,
                 lr:float=2e-4, betas:Tuple[float, float]=(0.5, 0.999),
                 device:str="cuda" if torch.cuda.is_available() else "cpu", ngpus:int=torch.cuda.device_count(), mixed:bool=False,
                 num_workers:int=4 if not torch.cuda.is_available() else 4 * torch.cuda.device_count(), 
                 distributed:bool=(torch.cuda.is_available() and torch.cuda.device_count() > 1),
                 seed:int=42, checkpointsdir:str='./CheckPoints', strtime:Optional[str]=None, **kwargs) -> None:

        self.setseed(seed)
        strtime = strtime or time.strftime('%Y_%m_%d_%H_%M_%S')
        print(f'seed was set manually to [{seed}]')
        print(f'start time: {strtime}')
        print(f'attached Devices: {device}, attached GPUs: {ngpus}')
        if distributed: print(f"Distrbuted Training on {ngpus} GPUs") 

        self.net = Network(backbone, dim, lr, betas, distributed, ngpus, mixed).to(device)

        self.trainloader = DataLoader(Dataset(datadir, source, mode, 'train', ismasked, augment), 
                                      batch_size=batch_size * (1 if not torch.cuda.is_available() else ngpus), 
                                      shuffle=True, 
                                      num_workers=num_workers, 
                                      pin_memory=True,
                                      drop_last=True)
        self.validloader = DataLoader(Dataset(datadir, source, mode, 'valid', ismasked, augment), 
                                      batch_size=batch_size * (1 if not torch.cuda.is_available() else ngpus), 
                                      shuffle=False, 
                                      num_workers=num_workers, 
                                      pin_memory=True,
                                      drop_last=True,)
        
        self.tensorboard = TensorBoard(Dataset(datadir, source, mode, 'test', ismasked, augment), 
                                       device, checkpointsdir, strtime, self.net.getName())
        
        self.checkpoint = CheckPoint(checkpointsdir, strtime)
        self.callback = Callback(monitor='val_loss', patience=10, min_delta=1e-5)
        self.logger = CSVLogger(checkpointsdir, strtime, self.net.getName())
        self.hyperparamters = Hyperparameters(checkpointsdir, strtime, self.net.getName())
        self.device = device
        self.metrics = {'loss' : Metric(), 'val_loss' : Metric()}
        
        self.hyperparamters.update({'class'     : self.net.getName(),
                                    'seed'      : seed,
                                    'datadir'   : datadir,
                                    'source'    : source,
                                    'mode'      : mode,
                                    'strtime'   : strtime,
                                    'dim'       : dim,
                                    'backbone'  : backbone,
                                    'mask'      : ismasked,
                                    'augment'   : augment,
                                    'batchsize' : batch_size,
                                    'lr'        : lr,
                                    'betas'     : betas,
                                    'mixed'     : mixed,
                                    })

    
    def setseed(self, seed:int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self) -> None:
        # Load the model from checkpoint
        self.net.load(self.checkpoint)

    def save(self) -> None:
        # Save current state of network to disk for later use
        self.net.save(self.checkpoint)

    def fit(self, epochs:int=100) -> None:
        for epoch in range(epochs):
            # training
            self.train(epoch + 1)

            # evaluating
            self.valid(epoch + 1)

            logs = {key : self.metrics[key].result() for key in self.metrics.keys()}

            self.net.scheduler.step(logs['val_loss'])

            self.logger.update(epoch, logs)
            self.callback.update(epoch, logs)
            
            print(f"Epoch: [{epoch+1:03d}/{epochs}] | {', '.join(f'{key} = {value:0.4f}' for key, value in logs.items())}")

            if self.callback.savecheckpoint:
                self.save()
                self.test(epoch + 1)

            for key in self.metrics.keys():
                self.metrics[key].reset() 


            if self.callback.stop_training:
                break
        
        self.logger.close()
        self.callback.close()

    def train(self, epoch:int) -> None:
        print(f'Training epoch: {str(epoch).zfill(3)}')
        progbar = ProgBar(len(self.trainloader), width=20, stateful_metrics=('epoch', 'loss'))
        self.net.train()
        

        for items in self.trainloader:
            ifgs, targets = self.cuda(*items)
            # train
            _, loss = self.net.process(ifgs, targets)

            logs = [
                   ('loss', self.metrics['loss'].update(loss)), 
            ]
           
            logs = [
                    ("epoch", epoch),
                ] + logs

            progbar.add(1, values=logs)
    
    def valid(self, epoch:int) -> None:
        print(f'Validating epoch: {str(epoch).zfill(3)}')
        progbar = ProgBar(len(self.validloader), width=20, stateful_metrics=('epoch', 'val_loss'))
        self.net.eval()
        

        for items in self.validloader:
            ifgs, targets = self.cuda(*items)
            # eval
            with torch.no_grad():
                _, loss = self.net.call(ifgs, targets)

            logs = [
                   ('val_loss', self.metrics['val_loss'].update(loss)), 
            ]
           
            logs = [
                    ("epoch", epoch),
                ] + logs

            progbar.add(1, values=logs)
    
    def test(self, epoch:int) -> None:
        print(f'Testing epoch: {str(epoch).zfill(3)}')
        self.tensorboard.create(epoch, self.net, save=True)


    def cuda(self, *args:torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (arg.to(self.device) for arg in args)
    
