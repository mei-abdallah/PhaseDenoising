from ..dtype import Data

class SlcData(Data):
    def plot(self, save:bool=False):
        return super().plot('Single Look Complex', 'RdBu_r', 'rad', save)