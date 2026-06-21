import torch
from abc import ABC, abstractmethod

class GenerateTask(ABC):
    def __init__(self) -> None:
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._info = None
        self.n_features = -1
        self.n_classes = None

        with torch.inference_mode():
            Xtr, ytr, Xte, yte, info = self._generate()

        self._X_train = Xtr
        self._y_train = ytr
        self._X_test = Xte
        self._y_test = yte
        self._info = info

    @abstractmethod
    def _generate(self):
        pass

    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def info(self):
        return self._info