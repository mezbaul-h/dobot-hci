# Fuzzy functions class, base for specified shapes


class FuzzyFunctions:
    def __init__(self) -> None:
        self._functions = {}  # Dictionary of functions
        self.peaks = {}  # centroids of each function

    # Get parameters of linear function aka y=ax+b
    def getLinearFunction(self, y1, y2, x1, x2):
        a = (y1 - y2) / (x1 - x2)
        b = y2 - a * x2
        return a, b

    def calculateOutput(self, value):  # Calculate output value of each function in functions dictionary
        output = {}
        for key in self._functions:
            output.update({key: self._functions[key](value)})
        return output

    def addFunction(self, key, function):  # Method for adding functions to dictionary
        self._functions.update({key: function})
        self.peaks.update({key: function.peak})
