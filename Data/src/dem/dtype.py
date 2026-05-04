from ..dtype import Data

class DemData(Data):
    def plot(self, save:bool=False):
        return super().plot('Dem', 'terrain', 'm', save)