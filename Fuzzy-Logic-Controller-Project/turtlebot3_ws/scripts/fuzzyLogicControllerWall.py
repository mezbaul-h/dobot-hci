from triangle import Triangle
from trapez import Trapez
from gauss import Gauss
from fuzzyFunctions import FuzzyFunctions


#import matplotlib.pyplot as plt
import numpy as np


# Fuzzy logic right wall folower class
class FuzzyLogicControllerWall:


    def __init__(self) -> None:

        #Initialization of fuzzy sets
        self.Xin = FuzzyFunctions()
        self.BRS = FuzzyFunctions()
        self.Xout = FuzzyFunctions()
        self.Yout = FuzzyFunctions()

        #Front right sensor fuzzy set
        self.Xin.addFunction("C", Trapez(-1.0, 1,1.0, 0.0))
        self.Xin.addFunction("M", Triangle(-0.5, 0.0, 0.5))
        self.Xin.addFunction("F", Trapez(0.0, 0.75, 1, 1))

        #Back right sensor fuzzy set
        self.Yin.addFunction("C", Trapez(0, 0, 0.25, 0.5))
        self.Yin.addFunction("M", Triangle(0.25, 0.5, 0.75))
        self.Yin.addFunction("F", Trapez(0.5, 0.75, 1, 1))

        #Output x
        self.Xout.addFunction("S", Triangle(0, 0.1, 0.2))
        self.Xout.addFunction("M", Triangle(0.2, 0.3, 0.4))
        self.Xout.addFunction("F", Triangle(0.4, 0.5, 0.6))

        #Output z
        self.Yout.addFunction("R", Triangle(-0.8, -0.5, -0.1))
        self.Yout.addFunction("F", Triangle(-0.1, 0, 0.1))
        self.Yout.addFunction("L", Triangle(0.1, 0.5, 0.8))

    #Main method to run all rules and return linear and angular velocity
    def rullBase(self, sensorFR, sensorBR):
        #self._plotSets()
        XinOut = self.Xin.calculateOutput(sensorFR)
        YinOut = self.Yin.calculateOutput(sensorBR)
        firingRulesX, firingRulesZ, firingValues = self._calculateRules(XinOut, YinOut)
        XoutOut, YoutOut = self._calculateCentroidOutputs(firingRulesX, firingRulesZ, firingValues)

        return XoutOut, YoutOut


    #Calculation of centroids outputs and final values
    def  _calculateCentroidOutputs(self, firingRulesX, firingRulesZ, firingValues):
        XoutOut = 0
        for value, key in zip(firingValues, firingRulesX):
            XoutOut += value*self.Xout.peaks[key]
        XoutOut = XoutOut/sum(firingValues)
        YoutOut = 0
        for value, key in zip(firingValues, firingRulesZ):
            YoutOut += value*self.Yout.peaks[key]
        YoutOut = YoutOut/sum(firingValues)

        return XoutOut, YoutOut


    #Defininition and calculation of rules and firing strenghts
    def _calculateRules(self, XinOut, YinOut):

        firingRulesX = []
        firingRulesZ = []
        firingValues = []

        for rule in XinOut:
            if(XinOut[rule] > 0):
                for rule2 in YinOut:
                    if(YinOut[rule2] > 0):
                        if(rule=='C' and rule2 == 'C'):
                            firingRulesX.append('S')
                            firingRulesZ.append('L')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])

                        elif(rule=='C' and rule2 == 'M'):
                            firingRulesX.append('S')
                            firingRulesZ.append('L')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='C' and rule2 == 'F'):
                            firingRulesX.append('S')
                            firingRulesZ.append('L')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='M' and rule2 == 'C'):
                            firingRulesX.append('M')
                            firingRulesZ.append('F')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='M' and rule2 == 'M'):
                            firingRulesX.append('M')
                            firingRulesZ.append('F')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='M' and rule2 == 'F'):
                            firingRulesX.append('M')
                            firingRulesZ.append('R')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='F' and rule2 == 'C'):
                            firingRulesX.append('F')
                            firingRulesZ.append('R')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='F' and rule2 == 'M'):
                            firingRulesX.append('F')
                            firingRulesZ.append('R')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])
                        elif(rule=='F' and rule2 == 'F'):
                            firingRulesX.append('F')
                            firingRulesZ.append('R')
                            if(XinOut[rule]<YinOut[rule2]):
                                firingValues.append(XinOut[rule])
                            else:
                                firingValues.append(YinOut[rule2])

        return firingRulesX, firingRulesZ, firingValues

    #Function for plotting memberships functions, must be commented on real robot tests
    '''def _plotSets(self):
        values = np.linspace(0, 1, 500)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xin.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Xin")
        plt.show()

        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yin.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Yin")
        plt.show()

        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["S"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Xout.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Xout")
        plt.show()

        values = np.linspace(-1, 1, 500)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["R"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Yout.calculateOutput(x)
            y.append(out["L"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Yout")
        plt.show()'''





def main():
    controller = FuzzyLogicControllerWall()
    print(controller.rullBase(0.45,0.45))



if __name__=="__main__":
    main()

