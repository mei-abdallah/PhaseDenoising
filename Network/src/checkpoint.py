import os
class CheckPoint():
    def __init__(self, dirname:str, strtime:str) -> None:
        self.dirname = dirname
        self.strtime = strtime
        self.checkdirs()

    def checkdirs(self) -> None:
        os.makedirs(f'{self.dirname}/{self.strtime}', exist_ok=True)

    def __str__(self) -> str:
        return f'{self.dirname}/{self.strtime}'