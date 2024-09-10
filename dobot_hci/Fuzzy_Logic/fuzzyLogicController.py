import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np

from .fuzzyFunctions import FuzzyFunctions
from .gauss import Gauss
from .trapez import Trapez
from .triangle import Triangle


# Fuzzy logic right wall folower class
class FuzzyLogicController:
    def __init__(self) -> None:
        # Initialization of fuzzy sets
        self.Xin = FuzzyFunctions()
        self.Yin = FuzzyFunctions()
        self.Xout = FuzzyFunctions()
        self.Yout = FuzzyFunctions()

        # Front right sensor fuzzy set
        self.Xin.addFunction("E", Trapez(-100.0, -100.0, -15.0, 0.0))
        self.Xin.addFunction("M", Triangle(-50.0, 0.0, 50.0))
        self.Xin.addFunction("W", Trapez(0.0, 15.0, 100.0, 100.0))

        # Back right sensor fuzzy set
        self.Yin.addFunction("N", Trapez(-100.0, -100.0, -50.0, 0.0))
        self.Yin.addFunction("M", Triangle(-50.0, 0.0, 50.0))
        self.Yin.addFunction("S", Trapez(0.0, 50.0, 100.0, 100.0))

        # Output x
        self.Xout.addFunction("E", Triangle(-10.0 * 4, -7.0 * 4, -2.0 * 4))
        self.Xout.addFunction("G", Triangle(-2.0 * 4, 0.0, 2.0 * 4))
        self.Xout.addFunction("W", Triangle(2.0 * 4, 7.0 * 4, 10.0 * 4))

        # Output z
        self.Yout.addFunction("N", Triangle(10.0 * 4, 7.0 * 4, 2.0 * 4))
        self.Yout.addFunction("G", Triangle(-2.0 * 4, 0.0, 2.0 * 4))
        self.Yout.addFunction("S", Triangle(-10.0 * 4, -7.0 * 4, -2.0 * 4))

    # Main method to run all rules and return linear and angular velocity
    def rullBase(self, sensorFR, sensorBR):
        # self._plotSets()
        XinOut = self.Xin.calculateOutput(sensorFR)
        YinOut = self.Yin.calculateOutput(sensorBR)
        firingRulesX, firingRulesZ, firingValues = self._calculateRules(XinOut, YinOut)
        XoutOut, YoutOut = self._calculateCentroidOutputs(firingRulesX, firingRulesZ, firingValues)

        return XoutOut, YoutOut

    # Calculation of centroids outputs and final values
    def _calculateCentroidOutputs(self, firingRulesX, firingRulesZ, firingValues):
        XoutOut = 0
        for value, key in zip(firingValues, firingRulesX):
            XoutOut += value * self.Xout.peaks[key]
        XoutOut = XoutOut / sum(firingValues)
        YoutOut = 0
        for value, key in zip(firingValues, firingRulesZ):
            YoutOut += value * self.Yout.peaks[key]
        YoutOut = YoutOut / sum(firingValues)

        return XoutOut, YoutOut

    # Defininition and calculation of rules and firing strenghts
    def _calculateRules(self, XinOut, YinOut):
        firingRulesX = []
        firingRulesZ = []
        firingValues = []

        for rule in XinOut:
            if XinOut[rule] > 0:
                for rule2 in YinOut:
                    if YinOut[rule2] > 0:
                        if rule == "E" and rule2 == "N":
                            firingRulesX.append("W")
                            firingRulesZ.append("S")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])

                        elif rule == "E" and rule2 == "S":
                            firingRulesX.append("W")
                            firingRulesZ.append("N")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "E" and rule2 == "M":
                            firingRulesX.append("W")
                            firingRulesZ.append("G")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "W" and rule2 == "N":
                            firingRulesX.append("E")
                            firingRulesZ.append("S")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "W" and rule2 == "S":
                            firingRulesX.append("E")
                            firingRulesZ.append("N")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "W" and rule2 == "N":
                            firingRulesX.append("E")
                            firingRulesZ.append("G")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "M" and rule2 == "N":
                            firingRulesX.append("G")
                            firingRulesZ.append("S")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "M" and rule2 == "S":
                            firingRulesX.append("G")
                            firingRulesZ.append("N")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif rule == "M" and rule2 == "M":
                            firingRulesX.append("G")
                            firingRulesZ.append("G")
                            if XinOut[rule] < YinOut[rule2]:
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])

        return firingRulesX, firingRulesZ, firingValues

    # Function for plotting memberships functions, must be commented on real robot tests
    def _plotSets(self):
        values = np.linspace(-1, 1, 500)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["E"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["W"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Xin")
        plt.show()

        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["N"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["S"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Yin")
        plt.show()

        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["W"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["G"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["E"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Xout")
        plt.show()

        values = np.linspace(-1, 1, 500)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["N"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["G"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["S"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Yout")
        plt.show()


def main():
    controller = FuzzyLogicController()
    print(controller.rullBase(0.45, 0.45))
    controller._plotSets()


if __name__ == "__main__":
    main()
