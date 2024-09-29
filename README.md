# Robot Leg Design and Optimization with PSO algorithm

Creating the robot leg, optimizing with PSO algorithm for 4bar mechanism serves as power transmitter.
This is a full-package developed for quadruped robots. The software is designed for (local) ordinary tasks with robots which are equipped with a mini processor.

<p align="center">
  <img src="https://github.com/Artinmi/Robot-Leg-4bar-PSO/blob/master/docs/leg.gif" width="45%" alt="Leg"/>
</p>


This project involves the design, simulation, and optimization of a robotic leg, using SolidWorks for mechanical design and the Particle Swarm Optimization (PSO) algorithm for parameter optimization. The robot leg was 3D printed, assembled, and controlled via an Arduino board for both inverse kinematics and manual control. The project uses Python's pyswarm library for PSO and Arduino for controlling servos.

## Table of Contents
1. [Introduction](#introduction)
2. [Design Process](#design-process)
3. [Optimization](#optimization)
4. [3D Printing](#3d-printing)
5. [Control System](#control-system)
6. [PSO Algorithm](#pso-algorithm)
7. [Arduino Control](#arduino-control)
8. [Installation](#installation)
9. [Usage](#usage)

## Introduction
The goal of this project was to design a functional robotic leg that could be optimized using the Particle Swarm Optimization (PSO) algorithm. The leg mechanism is a four-bar linkage system, designed for flexibility and robustness in movement. 

## Design Process
The robot leg was designed using SolidWorks, focusing on a four-bar linkage mechanism, which offers simplicity and effective force transmission. After modeling the components, they were evaluated and adjusted to meet the functional requirements.

- Link 1: Hip
- Link 2: Crank
- Link 3: Coupler
- Link 4: Knee

Once the design was completed, the components were 3D printed for further testing.

<p align="center">
  <img src="https://github.com/Artinmi/Robot-Leg-4bar-PSO/blob/master/working%20model%202-D/jump_working_model2D.gif" width="45%" alt="Leg"/>
</p>


## Optimization
We used the Particle Swarm Optimization (PSO) algorithm to optimize the dimensions of the leg's components for better performance. The optimization focused on four key lengths (L1, L2, L3, L4) to enhance the leg's stability and range of motion.

The PSO algorithm was implemented using the pyswarm library in Python. The algorithm adjusted the leg's linkage lengths to optimize its overall functionality.

### PSO Output
The optimized lengths of the leg components were:
- L1: 10 cm
- L2: 3.2 cm
- L3: 9.5 cm
- L4: 4.5 cm

## 3D Printing
After the design and optimization, the robot leg was 3D printed using Fused Deposition Modeling (FDM). The printed components were assembled and connected to servos for motion control.

<p align="center">
  <img src="https://github.com/Artinmi/Robot-Leg-4bar-PSO/blob/master/docs/render.png" width="55%" alt="Leg"/>
</p>

## Control System
The leg is controlled using two methods:
1. Inverse Kinematics: The leg is controlled automatically based on mathematical calculations.
2. Manual Control: A joystick connected to an Arduino is used for manual movement of the leg.

## PSO Algorithm
The PSO algorithm was implemented in Python to optimize the leg mechanism's lengths. The code can be found in the final_pso_velocity.ipynb file. The following steps outline the process:
1. Define the objective function: This function calculates the performance of the leg based on the lengths of the components.
2. Run the PSO algorithm: Using pyswarm, the PSO algorithm adjusts the lengths to minimize the objective function.
3. Visualize results: The final optimized leg is visualized in Python.

For more detailed explanations and the full code, refer to the ```.ipynb``` file.

## Arduino Control
The robot leg is controlled using an Arduino connected to servos. The leg_movement.ino file contains the code to control the leg's movement. The setup includes:
- Servo motors connected to the robot leg joints.
- Joystick for manual control.
- Arduino board for processing control signals.

<p align="center">
  <img src="https://github.com/Artinmi/Robot-Leg-4bar-PSO/blob/master/docs/Circuit.jpg" width="45%" alt="Leg"/>
</p>

## Installation
### Dependencies

This software is built on Python, which needs some libraries to be installed first, the Robot-Leg-4bar-PSO depends on following libraries:

- [numpy](https://numpy.org/install/) 
- [matplotlib](https://matplotlib.org/stable/install/index.html) 
- [pyswarm](https://pyswarms.readthedocs.io/en/latest/installation.html) 
- [joblib](https://joblib.readthedocs.io/en/latest/installing.html) 
### Installation
1. Clone this repository:

        git clone https://github.com/Artinmi/Robot-Leg-4bar-PSO.git
    
3. Install the required Python packages:

         ``` pip install pyswarm ```
    

5. Upload the Arduino code:
   - Open leg_movement.ino in the Arduino IDE.
   - Connect your Arduino board and upload the code.

<p align="center">
  <img src="https://github.com/Artinmi/Robot-Leg-4bar-PSO/blob/master/docs/robot%20leg.jpg" width="45%" alt="Leg"/>
</p>

## Usage
- Optimization: Run the PSO optimization script in Python to find the best dimensions for the leg.
        python  ```optimize_leg.py```
    
- Control: Use the Arduino joystick to manually control the leg or run inverse kinematics for automated movement.

## Credits
This project was developed for mechanism design project at IUST

### Contributions
Contributions are always welcome! If you'd like to improve the project or add new features:
1. Fork this repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request for review.

### Contact
If you have any questions or suggestions, feel free to reach out:

- Artin Mokhtariha - [artin1382mokhtariha@gmail.com](mailto:artin1382mokhtariha@gmail.com)
- GitHub: [Artinmi](https://github.com/Artinmi)
- Linkedin Post:
