import math

from fuzzyFunctions import FuzzyFunctions


# Class for gaussian function shape
class Gauss(FuzzyFunctions):
    def __init__(self, a, dev) -> None:
        self.a = a
        self.dev = dev

    def __call__(self, value):  # Overloaded call operator for output value calculation
        outVal = math.exp(-0.5 * ((value - self.a) / self.dev) ** 2)
        return outVal
