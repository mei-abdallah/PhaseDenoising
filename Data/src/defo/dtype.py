from ..dtype import Data

class DefoData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Defo', 'RdBu_r', 'rad', save)