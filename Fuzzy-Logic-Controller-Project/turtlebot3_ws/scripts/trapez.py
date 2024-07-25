from fuzzyFunctions import FuzzyFunctions


## Class for trapezoidal function shape
class Trapez(FuzzyFunctions):

    def __init__(self, begin, peak, peak2, end) -> None:
        self.begin = begin
        self.peak = peak
        self.peak2 = peak2
        self.end = end

    def __call__(self, value): #Overloaded call operator for outputs calculation
        if(value < self.peak):
            if(self.begin != self.peak):
                [a,b] = self.getLinearFunction(0,1,self.begin, self.peak)
                outVal = a*value+b
            else:
                outVal = 1
        elif(value >= self.peak and value <= self.peak2):
            outVal = 1
        elif(value > self.peak2 and value <= self.end):
            if(self.peak2 != self.end):
                [a,b] = self.getLinearFunction(0,1, self.end, self.peak2)
                outVal = a*value+b
            else:
                outVal = 1
        else:
            outVal = 0
        if outVal <0:
            outVal = 0
        return outVal
    
