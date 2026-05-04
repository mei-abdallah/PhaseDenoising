import torch

class Metric():
    def __init__(self) -> None:
        self.total = torch.tensor([0.0])
        self.count = torch.tensor([0.0])

    def update(self, values:torch.Tensor) -> float:
        values = values.detach()
        
        if not len(values.size()):
            values = torch.tensor([values.item()])
        
        self.count += values.size(0)
        self.total += values.sum(0).item()

        return values.mean().item()

    def result(self,)-> float:
        return (self.total/self.count).item()

    def reset(self,) -> None:
        self.total = torch.tensor([0.0])
        self.count = torch.tensor([0.0])
