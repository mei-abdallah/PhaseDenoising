from ..dtype import Data, MaskedData

class DelayData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Delay', 'RdBu_r', 'rad', save)
    
class DelayMaskedData(MaskedData):
    def plot(self, save:bool=False) -> None:
        return super().plot('Delay', 'RdBu_r', 'rad', save)
    