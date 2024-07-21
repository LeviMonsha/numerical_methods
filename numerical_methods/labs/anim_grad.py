import matplotlib.pyplot as plt
import numpy as np
import time

def f(x):
    return x*x - 5*x + 5

def df(x):
    return 2*x - 5

N = 20
xx = 0
lmb = 0.08

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()
fig, axis = plt.subplots()
axis.grid(True)

axis.plot(x_plt, f_plt)
point = axis.scatter(xx, f(xx), c="red")

for i in range(N):
    xx = xx - lmb * df(xx)

    point.set_offsets([xx, f(xx)])

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.12)

plt.ioff()
print(xx)
axis.scatter(xx, f(xx), c="blue")
plt.show()

# x = [1, 5, 10, 15, 20]
# y1 = [1, 7, 3, 5, 11]
# y2 = [i*1.2 + 1 for i in y1]
# y3 = [i*1.2 + 1 for i in y2]
# y4 = [i*1.2 + 1 for i in y3]
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 7))
# axs[0, 0].plot(x, y1, '-')
# axs[0, 1].plot(x, y2, '--')
# axs[1, 0].plot(x, y3, '-.')
# axs[1, 1].plot(x, y4, ':')
# fig.show()