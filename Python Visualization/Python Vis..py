# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# link lengths (in cm)
L1 = 9
L2 = 5
L3 = 9
L4 = 5
L5 = 2.5
M = 10
# angles
Alpha = 0  # estimated as 30 degree
Teta1 = -(np.pi / 4)  # arbirary

# time (per angle)
t = np.arange(0, np.pi, 0.001)

# angular velocity
W = 10  # (rad/s)
Teta2 = W * t

# Solution to Position Problem
# این قسمت در فایل پی دی اف توضیح داده شده است
# EQ.1
A = 2 * L1 * L4 * np.cos(Teta1) - 2 * L2 * L4 * np.cos(Teta2)
B = 2 * L1 * L4 * np.sin(Teta1) - 2 * L2 * L4 * np.sin(Teta2)
C = L1 ** 2 + L2 ** 2 + L4 ** 2 - L3 ** 2 - 2 * L1 * L2 * (
            np.cos(Teta1) * np.cos(Teta2) + np.sin(Teta1) * np.sin(Teta2))

# delta dor solving the EQ1
delta = 4 * B ** 2 - 4 * (C - A) * (C + A)
if np.any(delta > 0):
    t1 = (-B + np.sqrt(B ** 2 - C ** 2 + A ** 2)) / (C - A)
    t2 = (-B - np.sqrt(B ** 2 - C ** 2 + A ** 2)) / (C - A)

Teta4_1 = 2 * np.arctan(t1)
Teta4_2 = 2 * np.arctan(t2)

Sin1 = (L1 * np.sin(Teta1) + L4 * np.sin(Teta4_1) - L2 * np.sin(Teta2)) / L3
Sin2 = (L1 * np.sin(Teta1) + L4 * np.sin(Teta4_2) - L2 * np.sin(Teta2)) / L3

Cos1 = (L1 * np.cos(Teta1) + L4 * np.cos(Teta4_1) - L2 * np.cos(Teta2)) / L3
Cos2 = (L1 * np.cos(Teta1) + L4 * np.cos(Teta4_2) - L2 * np.cos(Teta2)) / L3

# finall Answers to EQ1
Teta3_1 = np.arctan2(Sin1, Cos1)
Teta3_2 = np.arctan2(Sin2, Cos2)

# Difining the indicator point to trace:
p1 = np.array([0, 0])
## add l5 cos sin
S = np.array([L2 * np.cos(Teta2) + L3 * np.cos(Teta3_2 + Alpha), L2 * np.sin(Teta2) + L3 * np.sin(Alpha + Teta3_2)])
Sx = S[0, :]
Sy = S[1, :]

Q = np.array([L2 * np.cos(Teta2), L2 * np.sin(Teta2)])

P = np.array([L1 * np.cos(Teta1) + L4 * np.cos(Teta4_2), L1 * np.sin(Teta1) + L4 * np.sin(Teta4_2)])

R = np.array([L1 * np.cos(Teta1), L1 * np.sin(Teta1)])

r = np.array([L1 * np.cos(Teta1) - L4 * np.cos(Teta4_2), L1 * np.sin(Teta1) - L4 * np.sin(Teta4_2)])
vx = np.diff(Sx) / np.diff(t)
vy = np.diff(Sy) / np.diff(t)
v = np.sqrt(vx ** 2 + vy ** 2)

ax = np.diff(vx)
ay = np.diff(vy)
a = np.sqrt(ax ** 2 + ay ** 2)
fig, ax = plt.subplots()
ax.set_xlim([-10, 15])
ax.set_ylim([-15, 10])
ax.grid(True)
#for j in range(len())
for i in range(len(t)):
    A_bar, = ax.plot([p1[0], Q[0, i]], [p1[1], Q[1, i]], color='k', linewidth=4)
    B_bar, = ax.plot([Q[0, i], P[0, i]], [Q[1, i], P[1, i]], color='k', linewidth=4)
    C_bar, = ax.plot([P[0, i], R[0]], [P[1, i], R[1]], color='k', linewidth=4)
    D_bar, = ax.plot([P[0, i], r[0, i]], [P[1, i], r[1, i]], color='k', linewidth=4)
    E_bar, = ax.plot([S[0, i], P[0, i]], [S[1, i], P[1, i]], color='k', linewidth=4)
    s_bar, = ax.plot([0, R[0]], [0, R[1]], color='k', linewidth=4)

    P5_circle = plt.Circle((r[:, i]), 0.08, color='b', fill=False)
    ax.add_artist(P5_circle)

    #stL2 = 'Q'
    #stL3 = 'P'

    #P2_text = ax.text(Q[0, i], Q[1, i] + 0.4, stL2, fontname='Cambria Math', fontsize=10)
    #P3_text = ax.text(P[0, i], P[1, i] + 0.4, stL3, fontname='Cambria Math', fontsize=10)

    plt.pause(0.03)

    if i < len(Teta2) - 1:
        A_bar.remove()
        B_bar.remove()
        C_bar.remove()
        D_bar.remove()
        E_bar.remove()
        #P3_text.remove()
        #P2_text.remove()






'''
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(v, linewidth=3)
plt.xlim([0, 100 * np.pi])
plt.ylim([0, 12])
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('velocity (m/s)')

plt.subplot(2, 1, 2)
plt.plot(a, linewidth=3)
plt.xlim([0, 100 * np.pi])
plt.ylim([0, 0.45])
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('acceleration (m/s^2)')

plt.figure()
plt.plot(Sx, linewidth=3)
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('X')

plt.figure()
plt.plot(Sy, linewidth=3)
plt.grid(True)
plt.xlabel('t(sec)')
plt.ylabel('Y')

plt.show()
'''


