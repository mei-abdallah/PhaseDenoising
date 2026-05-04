from ..dtype import Data


class NoiseData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Noise', 'RdBu_r', 'rad', save)