import numpy as np
from fuzzyFunctions import FuzzyFunctions
from gauss import Gauss
from trapez import Trapez
from triangle import Triangle

# import matplotlib.pyplot as plt


# Class for combination of two fuzzy logic controllers (right edge and obstacle avoidance)
class Combining:
    def __init__(self) -> None:
        # Initialization of fuzzy functions
        self.RWF = FuzzyFunctions()
        self.OA = FuzzyFunctions()

        # Right wall follower fuzzy set
        self.RWF.addFunction("C", Trapez(0.0, 0.75, 1, 1))

        # Obstacle avoidance fuzzy set
        self.OA.addFunction("C", Trapez(0, 0, 0.25, 0.5))

    # Calculation of combined outputs
    def combine(self, F, FR, FL, BR, xVelRWF, xVelOA, zVelRWF, zVelOA):
        OA_sensor = min(F, FR, FL)  # Minimum among sensors(front, front-right and front-left)
        RWF_sensor = min(FR, BR)  # Minimum between sensors(front-right, back-right)
        fuzzify_RWF = self.RWF.calculateOutput(RWF_sensor)
        fuzzify_OA = self.OA.calculateOutput(OA_sensor)

        xOut = (fuzzify_RWF["C"] * xVelRWF + fuzzify_OA["C"] * xVelOA) / (fuzzify_RWF["C"] + fuzzify_OA["C"])
        zOut = (fuzzify_RWF["C"] * zVelRWF + fuzzify_OA["C"] * zVelOA) / (fuzzify_RWF["C"] + fuzzify_OA["C"])

        return xOut, zOut

    # Method for plotting fuzzy set functions for controllers combination, must be commented on real robots
    """def plotSets(self):
        values = np.linspace(0, 1, 500)
        y = []
        for x in values:
            out = self.RWF.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        plt.grid()
        plt.title("RWF")
        plt.show()
    
        y = []
        for x in values:
            out = self.OA.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        plt.grid()
        plt.title("OA")
        plt.show()"""


def main():
    f = 1
    fr = 1
    fl = 1
    br = 1
    xVelRWF = 1
    xVelOA = 1
    zVelRWF = 1
    zVelOA = 1
    controller = Combining()
    print(controller.combine(f, fr, fl, br, xVelRWF, xVelOA, zVelRWF, zVelOA))
    # controller.plotSets()


if __name__ == "__main__":
    main()
