"""
Dobot(port, verbose=False) Creates an instance of dobot connected to given serial port.

port: string with name of serial port to connect
verbose: bool will print to console all serial comms
.pose() Returns the current pose of dobot, as a tuple (x, y, z, r, j1, j2, j3, j4)

x: float current x cartesian coordinate
y: float current y cartesian coordinate
z: float current z cartesian coordinate
r: float current effector rotation
j1: float current joint 1 angle
j2: float current joint 2 angle
j3: float current joint 3 angle
j4: float current joint 4 angle
.move_to(x, y, z, r, wait=False) queues a translation in dobot to given coordinates

x: float x cartesian coordinate to move
y: float y cartesian coordinate to move
z: float z cartesian coordinate to move
r: float r effector rotation
wait: bool waits until command has been executed to return to process
.speed(velocity, acceleration) changes velocity and acceleration at which the dobot moves to future coordinates

velocity: float desired translation velocity
acceleration: float desired translation acceleration
.suck(enable)

enable: bool enables/disables suction
.grip(enable)

enable: bool enables/disables gripper
"""
import time

import numpy as np
import serial
from pydobot import Dobot
from serial.tools import list_ports

from .Fuzzy_Logic.fuzzyLogicController import FuzzyLogicController
from .Fuzzy_Logic.type2 import RobotArmFuzzyController


class Movement:
    def __init__(self):
        available_ports = list_ports.comports()
        if not available_ports:
            raise ValueError("No serial ports found. Is the Dobot connected?")

        self.port = available_ports[0].device
        # print(self.port)
        self.port = "/dev/ttyUSB0"
        try:
            self.device = Dobot(port=self.port)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Dobot on port {self.port}: {str(e)}")

        self.position = self.device.pose()
        # self.position = [ 260.19921875, 0.0, 1.1702880859375, 0.0, 0.0, 42.107452392578125, 42.32732391357422, 0.0]
        print("Initial position: ", self.position)
        self.newPose = list(self.position)
        # self.controller = FuzzyLogicController()
        self.controller = RobotArmFuzzyController()

    def __exit__(self):
        self.device.close()

    def getTarget(self):
        pass

    def resetPosition(self):
        self.device.move_to(self.position[0], self.position[1], self.position[2], self.position[3], wait=True)

    def getPosition(self):
        pose = self.device.pose()
        print(pose)
        return pose

    def step(self, gripper, goal):
        print("Gripper: ", gripper, " Goal: ", goal)
        error = np.array(gripper) - np.array(goal)
        min_val = -100.0
        max_val = 100.0
        error = np.clip(error, min_val, max_val)

        # update = np.array(self.controller.rullBase(error[0], error[1]))
        update = np.array(self.controller.evaluate(error[0], error[1]))

        if update.all():
            self.newPose[0] = self.newPose[0] + update[1]
            self.newPose[1] = self.newPose[1] + update[0]

        print("Change ", update)

        try:
            self.device.move_to(self.newPose[0], self.newPose[1], self.newPose[2], self.newPose[3], wait=True)
        except AttributeError:
            pass

        if abs(update[0]) < 1 and abs(update[1]) < 1:
            return True
        else:
            return False

    def pick(self):
        try:
            self.device.move_to(self.newPose[0] + 20, self.newPose[1], -70.0, self.newPose[3], wait=True)
            self.device.suck(enable=True)
            # time.sleep(5)
            # self.device.move_to(self.position[0], self.position[1], self.position[2], self.position[3], wait=True)
            # self.newPose[0] = self.position [0]
            # self.newPose[1] = self.position[1]
            # time.sleep(5)
            # self.device.suck(enable=False)
        except AttributeError:
            pass

    def drop(self):
        self.device.move_to(self.newPose[0], self.newPose[1] + 40, -55, self.newPose[3], wait=True)
        self.device.suck(enable=False)
