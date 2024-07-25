from triangle import Triangle
from trapez import Trapez
from gauss import Gauss
from fuzzyFunctions import FuzzyFunctions


#import matplotlib.pyplot as plt
import numpy as np


# Fuzzy logic obstacle avoidance class
class FuzzyLogicControllerObstacle:

    def __init__(self) -> None:
        
        #Initialization of fuzzy sets functions
        self.FRS = FuzzyFunctions()
        self.FLS = FuzzyFunctions()
        self.FS = FuzzyFunctions()
        self.Velx = FuzzyFunctions()
        self.Velz = FuzzyFunctions()

        #Front right sensor fuzzy set
        self.FRS.addFunction("C", Trapez(0, 0, 0.25, 0.5))
        self.FRS.addFunction("M", Triangle(0.25, 0.5, 0.75))
        self.FRS.addFunction("F", Trapez(0.5, 0.75, 1, 1))

        #Front left sensor fuzzy set
        self.FLS.addFunction("C", Trapez(0, 0, 0.25, 0.5))
        self.FLS.addFunction("M", Triangle(0.25, 0.5, 0.75))
        self.FLS.addFunction("F", Trapez(0.5, 0.75, 1, 1))

        #Front sensor fuzzy set
        self.FS.addFunction("C", Trapez(0, 0, 0.25, 0.5))
        self.FS.addFunction("M", Triangle(0.25, 0.5, 0.75))
        self.FS.addFunction("F", Trapez(0.5, 0.75, 1, 1))

        #Output x
        self.Velx.addFunction("S", Triangle(-0.1, 0.0, 0.1))
        self.Velx.addFunction("M", Triangle(0.1, 0.2, 0.3))
        self.Velx.addFunction("F", Triangle(0.3, 0.4, 0.5))

        #Output z
        self.Velz.addFunction("R", Triangle(-0.8, -0.5, -0.1))
        self.Velz.addFunction("F", Triangle(-0.1, 0, 0.1))
        self.Velz.addFunction("L", Triangle(0.1, 0.5, 0.8))


    #Main method to run all rules and return linear and angular velocity
    def rullBase(self, sensorFR, sensorFL, sensorF):
        #self._plotSets()
        frsOut = self.FRS.calculateOutput(sensorFR)
        flsOut = self.FLS.calculateOutput(sensorFL)
        fsOut = self.FS.calculateOutput(sensorF)
        firingRulesX, firingRulesZ, firingValues = self._calculateRules(frsOut, flsOut, fsOut)
        velXOut, velZOut = self._calculateCentroidOutputs(firingRulesX, firingRulesZ, firingValues)

        return velXOut, velZOut
        
    #Calculation of centroids outputs and final values
    def  _calculateCentroidOutputs(self, firingRulesX, firingRulesZ, firingValues):
        velXOut = 0
        for value, key in zip(firingValues, firingRulesX):
            velXOut += value*self.Velx.peaks[key]
        velXOut = velXOut/sum(firingValues)
        velZOut = 0
        for value, key in zip(firingValues, firingRulesZ):
            velZOut += value*self.Velz.peaks[key]
        velZOut = velZOut/sum(firingValues)

        return velXOut, velZOut
        
    #Defininition and calculation of rules and firing strenghts
    def _calculateRules(self, frsOut, flsOut, fsOut):
        
        firingRulesX = []
        firingRulesZ = []
        firingValues = []
        for rule in frsOut:
            if(frsOut[rule] > 0):
                for rule2 in flsOut:
                    if(flsOut[rule2] > 0):
                        for rule3 in fsOut:
                            if(fsOut[rule3] > 0):
                                if(rule=='C' and rule2 == 'C' and rule3 == 'C'): #1
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'C' and rule3 == 'M'): #2
                                    firingRulesX.append('M')
                                    firingRulesZ.append('F')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'C' and rule3 == 'F'): #3
                                    firingRulesX.append('M')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'M' and rule3 == 'C'): #4 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'M' and rule3 == 'M'): #5
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'M' and rule3 == 'F'): #6
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'F' and rule3 == 'C'): #7
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'F' and rule3 == 'M'):#8
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='C' and rule2 == 'F' and rule3 == 'F'): #9
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                


                                elif(rule=='M' and rule2 == 'C' and rule3 == 'C'): #10 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'C' and rule3 == 'M'): #11
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'C' and rule3 == 'F'): #12
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'M' and rule3 == 'C'): #13 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'M' and rule3 == 'M'): #14 
                                    firingRulesX.append('M')
                                    firingRulesZ.append('F')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'M' and rule3 == 'F'): #15
                                    firingRulesX.append('M')
                                    firingRulesZ.append('F')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'F' and rule3 == 'C'): #16 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'F' and rule3 == 'M'):#17
                                    firingRulesX.append('M')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='M' and rule2 == 'F' and rule3 == 'F'): #18
                                    firingRulesX.append('F')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                

                                elif(rule=='F' and rule2 == 'C' and rule3 == 'C'): #19 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'C' and rule3 == 'M'): #20
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'C' and rule3 == 'F'): #21
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'M' and rule3 == 'C'): #22 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'M' and rule3 == 'M'): #23 
                                    firingRulesX.append('M')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'M' and rule3 == 'F'): #24
                                    firingRulesX.append('M')
                                    firingRulesZ.append('R')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'F' and rule3 == 'C'): #25 
                                    firingRulesX.append('S')
                                    firingRulesZ.append('L')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'F' and rule3 == 'M'):#26
                                    firingRulesX.append('M')
                                    firingRulesZ.append('F')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))

                                elif(rule=='F' and rule2 == 'F' and rule3 == 'F'): #27
                                    firingRulesX.append('F')
                                    firingRulesZ.append('F')
                                    firingValues.append(min(frsOut[rule], flsOut[rule2], fsOut[rule3]))


        return firingRulesX, firingRulesZ, firingValues
        
    #Function for plotting memberships functions, must be commented on real robot tests
    '''def _plotSets(self):
        values = np.linspace(0, 1, 500)
        y = []
        for x in values:
            out = self.FRS.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FRS.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FRS.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("FRS")
        plt.show()
    
        y = []
        for x in values:
            out = self.FLS.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FLS.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FLS.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("FLS")
        plt.show()

        y = []
        for x in values:
            out = self.FS.calculateOutput(x)
            y.append(out["C"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FS.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.FS.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("FS")
        plt.show()

        y = []
        for x in values:
            out = self.Velx.calculateOutput(x)
            y.append(out["S"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Velx.calculateOutput(x)
            y.append(out["M"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Velx.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Velx")
        plt.show()
       
        values = np.linspace(-1, 1, 500)
        y = []
        for x in values:
            out = self.Velz.calculateOutput(x)
            y.append(out["R"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Velz.calculateOutput(x)
            y.append(out["F"])
        plt.plot(values, y)
        y = []
        for x in values:
            out = self.Velz.calculateOutput(x)
            y.append(out["L"])
        plt.plot(values, y)
        plt.grid()
        plt.title("Velz")
        plt.show()'''

        



def main():
    controller = FuzzyLogicControllerObstacle()
    print(controller.rullBase(1, 1, 0.3237764537334442))



if __name__=="__main__":
    main()

        