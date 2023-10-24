SAMPLING_RATE = 250

class WaviDataset:
    def __init__(self, name, data) -> None:
        self.name = name
        self.data = data

    # def __name__(self):
    #     return self.__class__.