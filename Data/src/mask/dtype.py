from ..dtype import Data
class MaskData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Mask', 'gray', None, save)