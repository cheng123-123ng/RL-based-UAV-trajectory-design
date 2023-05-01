import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.grid()
ln2, = ax.plot([], [], '-', color='r', lw=2)

xdata, ydata = [], []
xdata1, ydata1 = [], []
ln1, = ax.plot([], [], 'r-')

theta = np.linspace(0, 2*np.pi, 100)
r_out = 1
r_in = 0.5

def init():
    ax.set_xlim(0, 7)
    ax.set_ylim(-2, 2)
    return ln1,ln2

def update(i):
    xdata1.append(i)
    ydata1.append(np.sin(i)*2)
    ln2.set_data(xdata1, ydata1)

    xdata.append(i)
    ydata.append(np.sin(i))
    ln1.set_data(xdata, ydata)
    return ln2,ln1,

ani = animation.FuncAnimation(fig=fig, func=update, frames=np.linspace(0, 2 * np.pi, 128), init_func=init, interval=1000)
ani.save('roll.gif', writer='imagemagick', fps=100)

plt.show()
