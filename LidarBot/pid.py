import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, thresh):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.desire_pos = 0
        self.desire_vel = 0
        self.thresh = thresh

    def reset(self):
        self.integral = 0

    def desire_position(self, pos, vel):
        self.desire_pos = pos
        self.desire_vel = vel

    def compute(self, pos, vel, dt):
        pos_error = self.desire_pos - pos
        vel_error = self.desire_vel - vel
        self.integral += pos_error * dt
        derivative = vel_error
        output = self.kp * pos_error + self.ki * self.integral + self.kd * derivative
        if abs(pos_error) < self.thresh:
            output = 0.0
        return output
