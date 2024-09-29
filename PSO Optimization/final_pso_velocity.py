#VElocity-based PSO


import numpy as np
from pyswarm import pso
from joblib import Parallel, delayed

# Define the four-bar mechanism with actual kinematic equations
class FourBarMechanism:
    def __init__(self, r1, r2, r3, r4, theta1):
        self.r1 = r1  # Ground link length
        self.r2 = r2  # Input link length
        self.r3 = r3  # Coupler link length
        self.r4 = r4  # Output link length
        self.theta1 = theta1  # Input angle
    def get_velocity(self, theta):
        r1, r2, r3, r4, theta1 = self.r1, self.r2, self.r3, self.r4, self.theta1
        t = np.arange(0, np.pi, 0.001)
        W = theta  # (rad/s)
        Theta2 = W * t
        A = 2 * r1 * r4 * np.cos(theta1) - 2 * r2 * r4 * np.cos(Theta2)
        B = 2 * r1 * r4 * np.sin(theta1) - 2 * r2 * r4 * np.sin(Theta2)
        C = r1 ** 2 + r2 ** 2 + r4 ** 2 - r3 ** 2 - 2 * r1 * r2 * (
            np.cos(theta1) * np.cos(Theta2) + np.sin(theta1) * np.sin(Theta2))
        delta = 4 * B ** 2 - 4 * (C - A) * (C + A)
        if np.any(delta < 0):
            return np.inf, np.inf
        theta4 = 2 * np.arctan2(-B + np.sqrt(delta), A + C)
        #theta3 = np.arctan2(r1 * np.sin(Theta2) - r4 * np.sin(theta4), r1 * np.cos(Theta2) - r4 * np.cos(theta4))
        x_4= r1 * np.sin(theta1) + r4 * np.sin(theta4)
        y_4= r1 * np.cos(theta1) + r4 * np.cos(theta4)
        dx_4_dt = np.gradient(x_4, t)
        dy_4_dt = np.gradient(y_4, t)
        velocity = np.sqrt(dx_4_dt**2 + dy_4_dt**2)
        return velocity[0]
    
    def get_position(self, theta):
        # Define the kinematic equations for the four-bar mechanism
        r1, r2, r3, r4, theta1 = self.r1, self.r2, self.r3, self.r4, self.theta1
        t = np.arange(0, np.pi, 0.001)
        W = theta  # (rad/s)
        Theta2 = W * t
        A = 2 * r1 * r4 * np.cos(theta1) - 2 * r2 * r4 * np.cos(Theta2)
        B = 2 * r1 * r4 * np.sin(theta1) - 2 * r2 * r4 * np.sin(Theta2)
        C = r1 ** 2 + r2 ** 2 + r4 ** 2 - r3 ** 2 - 2 * r1 * r2 * (
            np.cos(theta1) * np.cos(Theta2) + np.sin(theta1) * np.sin(Theta2))
        delta = 4 * B ** 2 - 4 * (C - A) * (C + A)
        if np.any(delta < 0):
            return np.inf, np.inf
        theta4 = 2 * np.arctan2(-B + np.sqrt(delta), A + C)
        theta3 = np.arctan2(r1 * np.sin(Theta2) - r4 * np.sin(theta4), r1 * np.cos(Theta2) - r4 * np.cos(theta4))
        x_coupler = r2 * np.cos(Theta2) + r3 * np.cos(theta3)
        y_coupler = r2 * np.sin(Theta2) + r3 * np.sin(theta3)
        return np.array([x_coupler[0], y_coupler[0]])




# Define the objective function to optimize
def objective_function(params):
    r1, r2, r3, r4 = params
    mechanism = FourBarMechanism(r1, r2, r3, r4, theta1=0)

    max_deviation = 0
    for i in np.linspace(0, 2 * np.pi, 100):
        velocity = mechanism.get_velocity(i)
        if np.isinf(velocity).any():
            return np.inf
        deviation = np.abs(velocity - target_velocity(i))
        max_deviation = max(max_deviation, deviation)

    return max_deviation

def target_velocity():
    # Assume a simple target velocity function for demonstration
    return 30  # constant target velocity

# Define bounds for the parameters (adjust as needed for your specific mechanism)
bounds = [(5, 10), (2, 4), (8, 10.5), (2, 4.5)]
# Perform PSO optimization
best_params, best_score = pso(objective_function, lb=[b[0] for b in bounds], ub=[b[1] for b in bounds], swarmsize=100)

print(f'Best Parameters: {best_params}')
print(f'Best Score: {best_score:0.3} ')