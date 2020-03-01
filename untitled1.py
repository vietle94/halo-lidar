# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 00:36:04 2020

@author: VIET
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
%matplotlib qt


class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        # self.mask = np.zeros(x.shape, dtype=bool)

        # self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)

        self.selector = RectangleSelector(ax, self, useblit=False,
                                          interactive=True)

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask


x, y = df.data['time'], df.data['range']
fig, ax = plt.subplots()
ax.pcolormesh(x, y, df.data['beta_raw'].transpose(), cmap='jet')
highlighter = Highlighter(ax, x, y)
highlighter.mask
plt.show()

selected_regions = highlighter.mask
print(selected_regions)
# Print the points _not_ selected
print(x[selected_regions], y[selected_regions])
plt.plot(x[selected_regions], y[selected_regions])


# %%


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


# %%
plt.clf()
plt.setp(plt.gca(), autoscale_on=False)

tellme('You will define a triangle, click to begin')

plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')
        pts = np.asarray(plt.ginput(3, timeout=-1))
        if len(pts) < 3:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break

    # Get rid of fill
    for p in ph:
        p.remove()
# %%
# Define a nice function of distance from individual pts


def f(x, y, pts):
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1/(np.sqrt((x - p[0])**2 + (y - p[1])**2))
    return 1/z


X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
Z = f(X, Y, pts)

CS = plt.contour(X, Y, Z, 20)

tellme('Use mouse to select contour label locations, middle button to finish')
CL = plt.clabel(CS, manual=True)

# %%
tellme('Now do a nested zoom, click to begin')
plt.waitforbuttonpress()

while True:
    tellme('Select two corners of zoom, middle mouse button to finish')
    pts = plt.ginput(2, timeout=-1)
    if len(pts) < 2:
        break
    (x0, y0), (x1, y1) = pts
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

tellme('All Done!')
plt.show()
