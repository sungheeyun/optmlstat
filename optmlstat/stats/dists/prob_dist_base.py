import abc


class ProbDistBase(abc.ABC):
    @property
    @abc.abstractmethod
    def num_variables(self) -> int:
        pass
