import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(u):
    x, y, vx, vy = u
    def Grav(a, b):
        return - M * G / np.sqrt(a**2 + b**2)**3 * a
    ax = Grav(x, y)
    ay = Grav(y, x)
    return np.array([vx, vy, ax, ay])

def TotE(v, x, y):
        return np.dot(v, v) / 2 - M * G / np.sqrt(x**2 + y**2)

def Euler(u):
    h = 0.001
    return u + h * f(u) 

def RungeKutta(u):
    h = 0.001
    
    k1 = f(u)
    k2 = f(u + h * k1 / 2)
    k3 = f(u + h * k2 / 2)
    k4 = f(u + h * k3)

    # first two of du/dt are last two of u, one step ago
    # last two are r-dependant

    u_nxt = u + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return u_nxt

def update(frame):
    global u1, u2
    global x1prev, y1prev, x2prev, y2prev
    global s1prev, s2prev

    u1 = RungeKutta(u1)
    u2 = Euler(u2)

    # x, y current vals

    x1, y1 = u1[0], u1[1]
    x1data.append(x1)
    y1data.append(y1)
    x2, y2 = u2[0], u2[1]
    x2data.append(x2)
    y2data.append(y2)

    line1.set_data(x1data, y1data)
    point1.set_data([x1], [y1])
    line2.set_data(x2data, y2data)
    point2.set_data([x2], [y2])

    # e1, e2 current vals
    
    v1 = u1[2:4]
    v2 = u2[2:4]

    e1 = TotE(v1, x1, y1)
    e2 = TotE(v2, x2, y2)

    def dr(x, y, xp, yp):
        return np.sqrt((x-xp)**2 + (y-yp)**2)

    s1 = s1prev + dr(x1, y1, x1prev, y1prev)
    s2 = s2prev + dr(x2, y2, x2prev, y2prev)

    path1.append(s1)
    path2.append(s2)

    E1data.append(e1)
    E2data.append(e2)

    E1.set_data(path1, E1data)
    E2.set_data(path2, E2data)

    x1prev = x1.copy()
    x2prev = x2.copy()
    y1prev = y1.copy()
    y2prev = y2.copy()
    s1prev = s1.copy()
    s2prev = s2.copy()

    return line1, point1, line2, point2, E1, E2

 
##### Initialize

x1prev = 10.0
x2prev = 10.0
y1prev = 5.0
y2prev = 5.0
s1prev = 0.0
s2prev = 0.0

u1 = np.array([10.0, 5.00, 10.0, 10.0])
u2 = u1.copy()
M = 20
G = 100

x1data, y1data = [], []
x2data, y2data = [], []
E1data, E2data = [], []
path1 = []
path2 = []

E0 = TotE(np.array([10.0, 10.0]), 10.0, 5.0)

##### Plotting

fig, ax0 = plt.subplots(1, 2, figsize = (7,5))
ax, ax1 = ax0 # Plot and Energies

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax1.set_xlim(0, 250)
ax1.set_ylim(0, 2 * E0)

ax.set_title('Keplerproblem Lösung')
ax1.set_title('Energiedifferenz')
ax1.set_ylabel('E/m')
ax1.set_xlabel('Zurückgelegte Strecke s')


ax.plot(0, 0, "ko")

# 1 = Rk, 2 = Euler
color1 = 'blue'
color2 = 'red'

line1, = ax.plot([], [], color = color1, lw=1)
point1, = ax.plot([], [], color = color1, marker = 'o')
line2, = ax.plot([], [], color = color2, lw=1)
point2, = ax.plot([], [], color = color2, marker = 'o')

E1, = ax1.plot([], [], color = color1, lw = 1, label = 'Runge Kutta Lösung')
E2, = ax1.plot([], [], color = color2, lw = 1, label = 'Euler Lösung')

ani = FuncAnimation(fig, update, frames=2000, interval=10, blit=True)

plt.legend()
plt.show()