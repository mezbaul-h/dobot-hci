from .fuzzyFunctions import FuzzyFunctions


# Class for Triangle function shape
class Triangle(FuzzyFunctions):
    def __init__(self, begin, peak, end) -> None:
        self.begin = begin
        self.peak = peak
        self.end = end

    def __call__(self, value):  # Overloaded call operator for calculation
        if value < self.peak:
            [a, b] = self.getLinearFunction(0, 1, self.begin, self.peak)
            outVal = a * value + b
        elif value == self.peak:
            outVal = 1
        elif value > self.peak and value <= self.end:
            [a, b] = self.getLinearFunction(0, 1, self.end, self.peak)
            outVal = a * value + b
        else:
            outVal = 0
        if outVal < 0:
            outVal = 0
        return outVal
