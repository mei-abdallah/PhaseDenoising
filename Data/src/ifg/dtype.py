from ..dtype import Data

class IfgData(Data):
    def plot(self, save:bool=False):
        return super().plot('Interferometric', 'RdBu_r', 'rad', save)