from abc import ABC, abstractmethod
import numpy as np


class ScoreCalculator(ABC):

    @abstractmethod
    def calculate_score(self):
        pass


class PassAtK(ScoreCalculator):

    def __str__(self) -> str:
        return "pass@k"

    @staticmethod
    def calculate_score(n, k, c):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        :return: pass@$k$ score"""
        
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


if __name__ == "__main__":
    sc = PassAtK()
    print(sc.calculate_score(10, 5, 3))
