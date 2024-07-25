import matplotlib.pyplot as plt
import numpy as np


def main():
    begin = 1
    endPoint = 4
    peak = 2
    peak2 = 3
    values = np.linspace(0, 11, 500)
    y = []

    

    for x in values:
        out = fuzzyFunctions.calculateOutput(x)
        y.append(out["C"]+ out["M"]+out["F"])
    plt.plot(values, y)
    plt.grid()
    plt.title("Fuzzy set")
    plt.show()


if __name__ == '__main__':
    main()