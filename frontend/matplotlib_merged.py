./subplots_axes_and_figures/subplot_demo.py
"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()




********************************************
./shapes_and_collections/path_patch_demo.py
"""
Demo of a PathPatch object.
"""
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

Path = mpath.Path
path_data = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
    ]
codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)
ax.add_patch(patch)

# plot control points and connecting lines
x, y = zip(*path.vertices)
line, = ax.plot(x, y, 'go-')

ax.grid()
ax.axis('equal')
plt.show()




********************************************
./shapes_and_collections/artist_reference.py
"""
Reference for matplotlib artists

This example displays several of matplotlib's graphics primitives (artists)
drawn using matplotlib API. A full list of artists and the documentation is
available at http://matplotlib.org/api/artist_api.html.

Copyright (c) 2010, Bartosz Telenczuk
BSD License
"""
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def label(xy, text):
    y = xy[1] - 0.15 # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


fig, ax = plt.subplots()
# create 3x3 grid to plot the artists
grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

patches = []

# add a circle
circle = mpatches.Circle(grid[0], 0.1,ec="none")
patches.append(circle)
label(grid[0], "Circle")

# add a rectangle
rect = mpatches.Rectangle(grid[1] - [0.025, 0.05], 0.05, 0.1, ec="none")
patches.append(rect)
label(grid[1], "Rectangle")

# add a wedge
wedge = mpatches.Wedge(grid[2], 0.1, 30, 270, ec="none")
patches.append(wedge)
label(grid[2], "Wedge")

# add a Polygon
polygon = mpatches.RegularPolygon(grid[3], 5, 0.1)
patches.append(polygon)
label(grid[3], "Polygon")

#add an ellipse
ellipse = mpatches.Ellipse(grid[4], 0.2, 0.1)
patches.append(ellipse)
label(grid[4], "Ellipse")

#add an arrow
arrow = mpatches.Arrow(grid[5, 0]-0.05, grid[5, 1]-0.05, 0.1, 0.1, width=0.1)
patches.append(arrow)
label(grid[5], "Arrow")

# add a path patch
Path = mpath.Path
path_data = [
     (Path.MOVETO, [ 0.018, -0.11 ]),
     (Path.CURVE4, [-0.031, -0.051]),
     (Path.CURVE4, [-0.115,  0.073]),
     (Path.CURVE4, [-0.03 ,  0.073]),
     (Path.LINETO, [-0.011,  0.039]),
     (Path.CURVE4, [ 0.043,  0.121]),
     (Path.CURVE4, [ 0.075, -0.005]),
     (Path.CURVE4, [ 0.035, -0.027]),
     (Path.CLOSEPOLY, [0.018, -0.11])
    ]
codes, verts = zip(*path_data)
path = mpath.Path(verts + grid[6], codes)
patch = mpatches.PathPatch(path)
patches.append(patch)
label(grid[6], "PathPatch")

# add a fancy box
fancybox = mpatches.FancyBboxPatch(
        grid[7] - [0.025, 0.05], 0.05, 0.1,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02))
patches.append(fancybox)
label(grid[7], "FancyBoxPatch")

# add a line
x,y = np.array([[-0.06, 0.0, 0.1], [0.05, -0.05, 0.05]])
line = mlines.Line2D(x + grid[8, 0], y + grid[8, 1], lw=5., alpha=0.3)
label(grid[8], "Line2D")

colors = np.linspace(0, 1, len(patches))
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
collection.set_array(np.array(colors))
ax.add_collection(collection)
ax.add_line(line)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.axis('equal')
plt.axis('off')

plt.show()




********************************************
./shapes_and_collections/scatter_demo.py
"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()




********************************************
./showcase/xkcd.py
import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Monroe
    # http://xkcd.com/418/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    ax.set_ylim([-30, 10])

    data = np.ones(100)
    data[70:] -= np.arange(30)

    plt.annotate(
        'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

    plt.plot(data)

    plt.xlabel('time')
    plt.ylabel('my overall health')
    fig.text(
        0.5, 0.05,
        '"Stove Ownership" from xkcd by Randall Monroe',
        ha='center')

    # Based on "The Data So Far" from XKCD by Randall Monroe
    # http://xkcd.com/373/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.bar([-0.125, 1.0-0.125], [0, 100], 0.25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([0, 1])
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([0, 110])
    ax.set_xticklabels(['CONFIRMED BY\nEXPERIMENT', 'REFUTED BY\nEXPERIMENT'])
    plt.yticks([])

    plt.title("CLAIMS OF SUPERNATURAL POWERS")

    fig.text(
        0.5, 0.05,
        '"The Data So Far" from xkcd by Randall Monroe',
        ha='center')

plt.show()




********************************************
./showcase/integral_demo.py
"""
Plot demonstrating the integral as the area under a curve.

Although this is a simple example, it demonstrates some important tweaks:

    * A simple line plot with custom color and line width.
    * A shaded region created using a Polygon patch.
    * A text label with mathtext rendering.
    * figtext calls to label the x- and y-axes.
    * Use of axis spines to hide the top and right spines.
    * Custom tick placement and labels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def func(x):
    return (x - 3) * (x - 5) * (x - 7) + 85


a, b = 2, 9 # integral limits
x = np.linspace(0, 10)
y = func(x)

fig, ax = plt.subplots()
plt.plot(x, y, 'r', linewidth=2)
plt.ylim(ymin=0)

# Make the shaded region
ix = np.linspace(a, b)
iy = func(ix)
verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
ax.add_patch(poly)

plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
         horizontalalignment='center', fontsize=20)

plt.figtext(0.9, 0.05, '$x$')
plt.figtext(0.1, 0.9, '$y$')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([])

plt.show()




********************************************
./mplot3d/mixed_subplots_demo.py
"""
Demonstrate the mixing of 2d and 3d subplots
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1,e1)


################
# First subplot
################
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

# Twice as tall as it is wide.
fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('A tale of 2 subplots')
ax = fig.add_subplot(2, 1, 1)
l = ax.plot(t1, f(t1), 'bo', 
            t2, f(t2), 'k--', markerfacecolor='green')
ax.grid(True)
ax.set_ylabel('Damped oscillation')


#################
# Second subplot
#################
ax = fig.add_subplot(2, 1, 2, projection='3d')
X = np.arange(-5, 5, 0.25)
xlen = len(X)
Y = np.arange(-5, 5, 0.25)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        linewidth=0, antialiased=False)

ax.set_zlim3d(-1, 1)

plt.show()





********************************************
./mplot3d/wire3d_animation_demo.py
from __future__ import print_function
"""
A very simple 'animation' of a 3D plot
"""
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time

def generate(X, Y, phi):
    R = 1 - np.sqrt(X**2 + Y**2)
    return np.cos(2 * np.pi * X + phi) * R

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)
Z = generate(X, Y, 0.0)

wframe = None
tstart = time.time()
for phi in np.linspace(0, 360 / 2 / np.pi, 100):

    oldcol = wframe

    Z = generate(X, Y, phi)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

    # Remove old line collection before drawing
    if oldcol is not None:
        ax.collections.remove(oldcol)

    plt.draw()

print ('FPS: %f' % (100 / (time.time() - tstart)))




********************************************
./mplot3d/pathpatch3d_demo.py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D


def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):

    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "y":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

p = Circle((5, 5), 3)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")


text3d(ax, (4, -2, 0), "X-axis", zdir="z", size=.5, usetex=False,
       ec="none", fc="k")
text3d(ax, (12, 4, 0), "Y-axis", zdir="z", size=.5, usetex=False,
       angle=.5*3.14159, ec="none", fc="k")
text3d(ax, (12, 10, 4), "Z-axis", zdir="y", size=.5, usetex=False,
       angle=.5*3.14159, ec="none", fc="k")

text3d(ax, (1, 5, 0),
       r"$\displaystyle G_{\mu\nu} + \Lambda g_{\mu\nu} = "
       r"\frac{8\pi G}{c^4} T_{\mu\nu}  $",
       zdir="z", size=1, usetex=True,
       ec="none", fc="k")

ax.set_xlim3d(0, 10)
ax.set_ylim3d(0, 10)
ax.set_zlim3d(0, 10)

plt.show()




********************************************
./mplot3d/surface3d_demo3.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
xlen = len(X)
Y = np.arange(-5, 5, 0.25)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

colortuple = ('y', 'b')
colors = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=False)

ax.set_zlim3d(-1, 1)
ax.w_zaxis.set_major_locator(LinearLocator(6))

plt.show()





********************************************
./mplot3d/lorenz_attractor.py
# Plot of the Lorenz Attractor based on Edward Lorenz's 1963 "Deterministic
# Nonperiodic Flow" publication.
# http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2
#
# Note: Because this is a simple non-linear ODE, it would be more easily
#       done using SciPy's ode solver, but this approach depends only
#       upon NumPy.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s=10, r=28, b=2.667) :
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
stepCnt = 10000

# Need one more for the initial values
xs = np.empty((stepCnt + 1,))
ys = np.empty((stepCnt + 1,))
zs = np.empty((stepCnt + 1,))

# Setting initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Stepping through "time".
for i in range(stepCnt) :
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()





********************************************
./mplot3d/offset_demo.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# This example demonstrates mplot3d's offset text display.
# As one rotates the 3D figure, the offsets should remain oriented
# same way as the axis label, and should also be located "away"
# from the center of the plot.
#
# This demo triggers the display of the offset text for the x and
# y axis by adding 1e5 to X and Y. Anything less would not
# automatically trigger it.

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.mgrid[0:6*np.pi:0.25, 0:4*np.pi:0.25]
Z = np.sqrt(np.abs(np.cos(X) + np.cos(Y)))

surf = ax.plot_surface(X + 1e5, Y + 1e5, Z, cmap='autumn', cstride=2, rstride=2)
ax.set_xlabel("X-Label")
ax.set_ylabel("Y-Label")
ax.set_zlabel("Z-Label")
ax.set_zlim(0, 2)

plt.show()





********************************************
./mplot3d/text3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
xs = (1, 4, 4, 9, 4, 1)
ys = (2, 5, 8, 10, 1, 2)
zs = (10, 3, 8, 9, 1, 8)

for zdir, x, y, z in zip(zdirs, xs, ys, zs):
    label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    ax.text(x, y, z, label, zdir)

ax.text(9, 0, 0, "red", color='red')
ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

ax.set_xlim3d(0, 10)
ax.set_ylim3d(0, 10)
ax.set_zlim3d(0, 10)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()





********************************************
./mplot3d/contourf3d_demo.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()





********************************************
./mplot3d/scatter3d_demo.py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()





********************************************
./mplot3d/wire3d_demo.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()





********************************************
./mplot3d/lines3d_demo.py
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()





********************************************
./mplot3d/polys3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

xs = np.arange(0, 10, 0.4)
verts = []
zs = [0.0, 1.0, 2.0, 3.0]
for z in zs:
    ys = np.random.rand(len(xs))
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                           cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 10)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 4)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)

plt.show()





********************************************
./mplot3d/surface3d_radial_demo.py
# By Armin Moser

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
step = 0.04
maxval = 1.0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create supporting points in polar coordinates
r = np.linspace(0,1.25,50)
p = np.linspace(0,2*np.pi,50)
R,P = np.meshgrid(r,p)
# transform them to cartesian system
X,Y = R*np.cos(P),R*np.sin(P)

Z = ((R**2 - 1)**2)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
ax.set_zlim3d(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')
plt.show()




********************************************
./mplot3d/contourf3d_demo2.py
"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()





********************************************
./mplot3d/contour3d_demo.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()





********************************************
./mplot3d/subplot3d_demo.py
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt


# imports specific to the plots in this example
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

# Twice as wide as it is tall.
fig = plt.figure(figsize=plt.figaspect(0.5))

#---- First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=10)

#---- Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()





********************************************
./mplot3d/rotate_axes3d_demo.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()





********************************************
./mplot3d/trisurf3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

n_angles = 36
n_radii = 8

# An array of radii
# Does not include radius r=0, this is to eliminate duplicate points
radii = np.linspace(0.125, 1.0, n_radii)

# An array of angles
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius
angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords
# (0, 0) is added here. There are no duplicate points in the (x, y) plane
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Pringle surface
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

plt.show()




********************************************
./mplot3d/contour3d_demo3.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()





********************************************
./mplot3d/bars3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
    xs = np.arange(20)
    ys = np.random.rand(20)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()





********************************************
./mplot3d/2dcollections3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='zs=0, zdir=z')

colors = ('r', 'g', 'b', 'k')
for c in colors:
    x = np.random.sample(20)
    y = np.random.sample(20)
    ax.scatter(x, y, 0, zdir='y', c=c)

ax.legend()
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

plt.show()





********************************************
./mplot3d/trisurf3d_demo2.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# u, v are parameterisation variables
u = (np.linspace(0, 2.0 * np.pi, endpoint=True, num=50) * np.ones((10, 1))).flatten()
v = np.repeat(np.linspace(-0.5, 0.5, endpoint=True, num=10), repeats=50).flatten()

# This is the Mobius mapping, taking a u, v pair and returning an x, y, z
# triple
x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(u, v)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

# The triangles in parameter space determine which x, y, z points are
# connected by an edge
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)

ax.set_zlim(-1, 1)

# First create the x and y coordinates of the points.
n_angles = 36
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)
angles[:,1::2] += np.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
z = (np.cos(radii)*np.cos(angles*3.0)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = mtri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

# tripcolor plot.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap)
plt.show()




********************************************
./mplot3d/contour3d_demo2.py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()





********************************************
./mplot3d/hist3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

plt.show()





********************************************
./mplot3d/surface3d_demo2.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

plt.show()





********************************************
./mplot3d/surface3d_demo.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()





********************************************
./animation/dynamic_image2.py
#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    repeat_delay=1000)

#ani.save('dynamic_images.mp4')


plt.show()




********************************************
./animation/basic_example_writer.py
# Same as basic_example, but writes files using a single MovieWriter instance
# without putting on screen
# -*- noplot -*-
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_line(num, data, line):
    line.set_data(data[...,:num])
    return line,

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
    interval=50, blit=True)
line_ani.save('lines.mp4', writer=writer)

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
    blit=True)
im_ani.save('im.mp4', writer=writer)




********************************************
./animation/random_data.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0, 1)

def update(data):
    line.set_ydata(data)
    return line,

def data_gen():
    while True: yield np.random.rand(10)

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
plt.show()




********************************************
./animation/subplots.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# This example uses subclassing, but there is no reason that the proper function
# couldn't be set up and then use FuncAnimation. The code is long, but not
# really complex. The length is due solely to the fact that there are a total
# of 9 lines that need to be changed for the animation as well as 3 subplots
# that need initial set up.
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)

        self.t = np.linspace(0, 80, 400)
        self.x = np.cos(2 * np.pi * self.t / 10.)
        self.y = np.sin(2 * np.pi * self.t / 10.)
        self.z = 10 * self.t

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        self.line1 = Line2D([], [], color='black')
        self.line1a = Line2D([], [], color='red', linewidth=2)
        self.line1e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal', 'datalim')

        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        self.line2 = Line2D([], [], color='black')
        self.line2a = Line2D([], [], color='red', linewidth=2)
        self.line2e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(0, 800)

        ax3.set_xlabel('x')
        ax3.set_ylabel('z')
        self.line3 = Line2D([], [], color='black')
        self.line3a = Line2D([], [], color='red', linewidth=2)
        self.line3e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax3.add_line(self.line3)
        ax3.add_line(self.line3a)
        ax3.add_line(self.line3e)
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(0, 800)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_len = 10
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])

        self.line2.set_data(self.y[:i], self.z[:i])
        self.line2a.set_data(self.y[head_slice], self.z[head_slice])
        self.line2e.set_data(self.y[head], self.z[head])

        self.line3.set_data(self.x[:i], self.z[:i])
        self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        self.line3e.set_data(self.x[head], self.z[head])

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e,
            self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines =  [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e,
            self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])

ani = SubplotAnimation()
#ani.save('test_sub.mp4')
plt.show()




********************************************
./animation/moviewriter.py
# This example uses a MovieWriter directly to grab individual frames and
# write them to a file. This avoids any event loop integration, but has
# the advantage of working with even the Agg backend. This is not recommended
# for use in an interactive setting.
# -*- noplot -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0,y0 = 0, 0

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()





********************************************
./animation/dynamic_image.py
#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = plt.imshow(f(x, y), cmap=plt.get_cmap('jet'))

def updatefig(*args):
    global x,y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x,y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()




********************************************
./animation/simple_anim.py
"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)        # x-array
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10.0))  # update the data
    return line,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    interval=25, blit=True)
plt.show()




********************************************
./animation/simple_3danim.py
"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def Gen_RandLine(length, dims=2) :
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length) :
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index-1] + step

    return lineData

def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_RandLine(25, 3) for index in range(50)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                              interval=50, blit=False)

plt.show()




********************************************
./animation/basic_example.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_line(num, data, line):
    line.set_data(data[...,:num])
    return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
    interval=50, blit=True)
#line_ani.save('lines.mp4')

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
    blit=True)
#im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()




********************************************
./animation/histogram.py
"""
This example shows how to use a path patch to draw a bunch of
rectangles for an animated histogram
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

fig, ax = plt.subplots()

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts = nrects*(1+3+1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5,0] = left
verts[0::5,1] = bottom
verts[1::5,0] = left
verts[1::5,1] = top
verts[2::5,0] = right
verts[2::5,1] = top
verts[3::5,0] = right
verts[3::5,1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

def animate(i):
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5,1] = top
    verts[2::5,1] = top

ani = animation.FuncAnimation(fig, animate, 100, repeat=False)
plt.show()




********************************************
./animation/bayes_update.py
# update a distribution based on new data.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.animation import FuncAnimation

class UpdateDist(object):
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = ss.beta.pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, ud, frames=np.arange(100), init_func=ud.init,
        interval=100, blit=True)
plt.show()




********************************************
./animation/animate_decay.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def data_gen():
    t = data_gen.t
    cnt = 0
    while cnt < 1000:
        cnt+=1
        t += 0.05
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)
data_gen.t = 0

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 5)
ax.grid()
xdata, ydata = [], []
def run(data):
    # update the data
    t,y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
    repeat=False)
plt.show()




********************************************
./animation/old_animation/draggable_legend.py
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3], label="test")

l = ax.legend()
d1 = l.draggable()

xy = 1, 2
txt = ax.annotate("Test", xy, xytext=(-30, 30),
                  textcoords="offset points",
                  bbox=dict(boxstyle="round",fc=(0.2, 1, 1)),
                  arrowprops=dict(arrowstyle="->"))
d2 = txt.draggable()


from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fn = get_sample_data("ada.png", asfileobj=False)
arr_ada = read_png(fn)

imagebox = OffsetImage(arr_ada, zoom=0.2)

ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )


ax.add_artist(ab)

d3 = ab.draggable(use_blit=True)
    
    
plt.show()




********************************************
./animation/old_animation/histogram_tkagg.py
"""
This example shows how to use a path patch to draw a bunch of
rectangles for an animated histogram
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

fig, ax = plt.subplots()

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts = nrects*(1+3+1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5,0] = left
verts[0::5,1] = bottom
verts[1::5,0] = left
verts[1::5,1] = top
verts[2::5,0] = right
verts[2::5,1] = top
verts[3::5,0] = right
verts[3::5,1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

def animate():
    if animate.cnt>=100:
        return

    animate.cnt += 1
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5,1] = top
    verts[2::5,1] = top
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, animate)
animate.cnt = 0
fig.canvas.manager.window.after(100, animate)
plt.show()




********************************************
./animation/old_animation/simple_anim_tkagg.py
#!/usr/bin/env python

"""
A simple example of an animated plot in tkagg
"""
from __future__ import print_function
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

def animate():
    tstart = time.time()                   # for profiling
    x = np.arange(0, 2*np.pi, 0.01)        # x-array
    line, = ax.plot(x, np.sin(x))

    for i in np.arange(1,200):
        line.set_ydata(np.sin(x+i/10.0))          # update the data
        fig.canvas.draw()                         # redraw the canvas
    print('FPS:' , 200/(time.time()-tstart))

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()




********************************************
./animation/old_animation/dynamic_image_gtkagg.py
#!/usr/bin/env python

from __future__ import print_function
"""
An animated image
"""
import time

import gobject
import gtk

import matplotlib
matplotlib.use('GTKAgg')

from pylab import *

fig = figure(1)
a = subplot(111)
x = arange(120.0)*2*pi/120.0
x = resize(x, (100,120))
y = arange(100.0)*2*pi/100.0
y = resize(y, (120,100))
y = transpose(y)
z = sin(x) + cos(y)
im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')

manager = get_current_fig_manager()
cnt = 0
tstart = time.time()
def updatefig(*args):
    global x, y, cnt, start
    x += pi/15
    y += pi/20
    z = sin(x) + cos(y)
    im.set_array(z)
    manager.canvas.draw()
    cnt += 1
    if cnt==50:
        print('FPS', cnt/(time.time() - tstart))
        return False
    return True

cnt = 0

gobject.idle_add(updatefig)
show()




********************************************
./animation/old_animation/animate_decay_tk_blit.py
from __future__ import print_function
import time, sys
import numpy as np
import matplotlib.pyplot as plt


def data_gen():
    t = data_gen.t
    data_gen.t += 0.05
    return np.sin(2*np.pi*t) * np.exp(-t/10.)
data_gen.t = 0

fig, ax = plt.subplots()
line, = ax.plot([], [], animated=True, lw=2)
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 5)
ax.grid()
xdata, ydata = [], []
def run(*args):
    background = fig.canvas.copy_from_bbox(ax.bbox)
    # for profiling
    tstart = time.time()

    while 1:
        # restore the clean slate background
        fig.canvas.restore_region(background)
        # update the data
        t = data_gen.t
        y = data_gen()
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()
        if t>=xmax:
            ax.set_xlim(xmin, 2*xmax)
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(ax.bbox)

        line.set_data(xdata, ydata)

        # just draw the animated artist
        ax.draw_artist(line)
        # just redraw the axes rectangle
        fig.canvas.blit(ax.bbox)

        if run.cnt==1000:
            # print the timing info and quit
            print('FPS:' , 1000/(time.time()-tstart))
            sys.exit()

        run.cnt += 1
run.cnt = 0



manager = plt.get_current_fig_manager()
manager.window.after(100, run)

plt.show()




********************************************
./animation/old_animation/animation_blit_wx.py
# For detailed comments on animation and the techniqes used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation

from __future__ import print_function

# The number of blits() to make before exiting
NBLITS = 1000

import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

import wx
import sys
import pylab as p
import numpy as npy
import time


# allow the user to disable the WXAgg accelerator from the command line
if '--no-accel' in sys.argv:
    import matplotlib.backends.backend_wxagg
    matplotlib.backends.backend_wxagg._use_accelerator(False)


fig, ax = plt.subplots()
canvas = fig.canvas


p.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
p.grid() # to ensure proper background restore

# create the initial line
x = npy.arange(0,2*npy.pi,0.01)
line, = p.plot(x, npy.sin(x), animated=True, lw=2)

# for profiling
tstart = time.time()
blit_time = 0.0

def update_line(*args):
    global blit_time

    if update_line.background is None:
        update_line.background = canvas.copy_from_bbox(ax.bbox)

    # restore the clean slate background
    canvas.restore_region(update_line.background)
    # update the data
    line.set_ydata(npy.sin(x+update_line.cnt/10.0))
    # just draw the animated artist
    ax.draw_artist(line)
    # just redraw the axes rectangle

    t = time.time()
    canvas.blit(ax.bbox)
    blit_time += time.time() - t

    if update_line.cnt == NBLITS:
        # print the timing info and quit
        frame_time = time.time() - tstart
        print('%d frames: %.2f seconds' % (NBLITS, frame_time))
        print('%d blits:  %.2f seconds' % (NBLITS, blit_time))
        print()
        print('FPS: %.2f' % (NBLITS/frame_time))
        print('BPS: %.2f' % (NBLITS/blit_time))
        sys.exit()

    update_line.cnt += 1
    wx.WakeUpIdle()



update_line.cnt = 0
update_line.background = None
wx.EVT_IDLE(wx.GetApp(), update_line)
p.show()




********************************************
./animation/old_animation/dynamic_image_wxagg2.py
#!/usr/bin/env python
"""
Copyright (C) 2003-2005 Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

"""
import sys, time, os, gc

import matplotlib
matplotlib.use('WXAgg')

from matplotlib import rcParams
import numpy as npy

import matplotlib.cm as cm

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.figure import Figure
from wx import *


TIMER_ID = NewId()

class PlotFigure(Frame):

    def __init__(self):
        Frame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        # On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(Size(fw, th))

        # Create a figure manager to manage things

        # Now put all into a sizer
        sizer = BoxSizer(VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, LEFT|TOP|GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, GROW)
        self.SetSizer(sizer)
        self.Fit()
        EVT_TIMER(self, TIMER_ID, self.onTimer)

    def init_plot_data(self):
        # jdh you can add a subplot directly from the fig rather than
        # the fig manager
        a = self.fig.add_axes([0.075,0.1,0.75,0.85])
        cax = self.fig.add_axes([0.85,0.1,0.075,0.85])
        self.x = npy.empty((120,120))
        self.x.flat = npy.arange(120.0)*2*npy.pi/120.0
        self.y = npy.empty((120,120))
        self.y.flat = npy.arange(120.0)*2*npy.pi/100.0
        self.y = npy.transpose(self.y)
        z = npy.sin(self.x) + npy.cos(self.y)
        self.im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')
        self.fig.colorbar(self.im,cax=cax,orientation='vertical')

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def onTimer(self, evt):
        self.x += npy.pi/15
        self.y += npy.pi/20
        z = npy.sin(self.x) + npy.cos(self.y)
        self.im.set_array(z)
        self.canvas.draw()
        #self.canvas.gui_repaint()  # jdh wxagg_draw calls this already

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

if __name__ == '__main__':
    app = PySimpleApp()
    frame = PlotFigure()
    frame.init_plot_data()

    # Initialise the timer - wxPython requires this to be connected to
    # the receiving event handler
    t = Timer(frame, TIMER_ID)
    t.Start(200)

    frame.Show()
    app.MainLoop()





********************************************
./animation/old_animation/simple_timer_wx.py
from __future__ import print_function
"""
A simple example of an animated plot using a wx backend
"""
import numpy as np
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.arange(0, 2*np.pi, 0.1)
line, = ax.plot(t, np.sin(t))
dt = 0.05


def update_line(event):
    if update_line.i==200:
        return False
    print('update', update_line.i)
    line.set_ydata(np.sin(t+update_line.i/10.))
    fig.canvas.draw()                 # redraw the canvas
    update_line.i += 1
update_line.i = 0

import wx
id = wx.NewId()
actor = fig.canvas.manager.frame
timer = wx.Timer(actor, id=id)
timer.Start(100)
wx.EVT_TIMER(actor, id, update_line)
#actor.Bind(wx.EVT_TIMER, update_line, id=id)

plt.show()




********************************************
./animation/old_animation/simple_anim_gtk.py
"""
A simple example of an animated plot using a gtk backend
"""
from __future__ import print_function
import time
import numpy as np
import matplotlib
matplotlib.use('GTKAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def animate():
    tstart = time.time()                 # for profiling
    x = np.arange(0, 2*np.pi, 0.01)        # x-array
    line, = ax.plot(x, np.sin(x))

    for i in np.arange(1,200):
        line.set_ydata(np.sin(x+i/10.0))  # update the data
        fig.canvas.draw()                         # redraw the canvas
    print('FPS:' , 200/(time.time()-tstart))
    raise SystemExit

import gobject
print('adding idle')
gobject.idle_add(animate)
print('showing')
plt.show()




********************************************
./animation/old_animation/dynamic_collection.py
import random
from matplotlib.collections import RegularPolyCollection
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show
from numpy.random import rand

fig = figure()
ax = fig.add_subplot(111, xlim=(0,1), ylim=(0,1), autoscale_on=False)
ax.set_title("Press 'a' to add a point, 'd' to delete one")
# a single point
offsets = [(0.5,0.5)]
facecolors = [cm.jet(0.5)]

collection = RegularPolyCollection(
    #fig.dpi,
    5, # a pentagon
    rotation=0,
    sizes=(50,),
    facecolors = facecolors,
    edgecolors = 'black',
    linewidths = (1,),
    offsets = offsets,
    transOffset = ax.transData,
    )

ax.add_collection(collection)

def onpress(event):
    """
    press 'a' to add a random point from the collection, 'd' to delete one
    """
    if event.key=='a':
        x,y = rand(2)
        color = cm.jet(rand())
        offsets.append((x,y))
        facecolors.append(color)
        collection.set_offsets(offsets)
        collection.set_facecolors(facecolors)
        fig.canvas.draw()
    elif event.key=='d':
        N = len(offsets)
        if N>0:
            ind = random.randint(0,N-1)
            offsets.pop(ind)
            facecolors.pop(ind)
            collection.set_offsets(offsets)
            collection.set_facecolors(facecolors)
            fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', onpress)

show()




********************************************
./animation/old_animation/movie_demo.py
#!/usr/bin/python
#
# Josh Lifton 2004
#
# Permission is hereby granted to use and abuse this document
# so long as proper attribution is given.
#
# This Python script demonstrates how to use the numarray package
# to generate and handle large arrays of data and how to use the
# matplotlib package to generate plots from the data and then save
# those plots as images.  These images are then stitched together
# by Mencoder to create a movie of the plotted data.  This script
# is for demonstration purposes only and is not intended to be
# for general use.  In particular, you will likely need to modify
# the script to suit your own needs.
#

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # For plotting graphs.
import numpy as np
import subprocess                 # For issuing commands to the OS.
import os
import sys                        # For determining the Python version.

#
# Print the version information for the machine, OS,
# Python interpreter, and matplotlib.  The version of
# Mencoder is printed when it is called.
#
print('Executing on', os.uname())
print('Python version', sys.version)
print('matplotlib version', matplotlib.__version__)

not_found_msg = """
The mencoder command was not found;
mencoder is used by this script to make an avi file from a set of pngs.
It is typically not installed by default on linux distros because of
legal restrictions, but it is widely available.
"""

try:
    subprocess.check_call(['mencoder'])
except subprocess.CalledProcessError:
    print("mencoder command was found")
    pass # mencoder is found, but returns non-zero exit as expected
    # This is a quick and dirty check; it leaves some spurious output
    # for the user to puzzle over.
except OSError:
    print(not_found_msg)
    sys.exit("quitting\n")


#
# First, let's create some data to work with.  In this example
# we'll use a normalized Gaussian waveform whose mean and
# standard deviation both increase linearly with time.  Such a
# waveform can be thought of as a propagating system that loses
# coherence over time, as might happen to the probability
# distribution of a clock subjected to independent, identically
# distributed Gaussian noise at each time step.
#

print('Initializing data set...')   # Let the user know what's happening.

# Initialize variables needed to create and store the example data set.
numberOfTimeSteps = 100   # Number of frames we want in the movie.
x = np.arange(-10,10,0.01)   # Values to be plotted on the x-axis.
mean = -6                 # Initial mean of the Gaussian.
stddev = 0.2              # Initial standard deviation.
meaninc = 0.1             # Mean increment.
stddevinc = 0.1           # Standard deviation increment.

# Create an array of zeros and fill it with the example data.
y = np.zeros((numberOfTimeSteps,len(x)), float)
for i in range(numberOfTimeSteps) :
    y[i] = (1/np.sqrt(2*np.pi*stddev))*np.exp(-((x-mean)**2)/(2*stddev))
    mean = mean + meaninc
    stddev = stddev + stddevinc

print('Done.')                       # Let the user know what's happening.

#
# Now that we have an example data set (x,y) to work with, we can
# start graphing it and saving the images.
#

for i in range(len(y)) :
    #
    # The next four lines are just like MATLAB.
    #
    plt.plot(x,y[i],'b.')
    plt.axis((x[0],x[-1],-0.25,1))
    plt.xlabel('time (ms)')
    plt.ylabel('probability density function')

    #
    # Notice the use of LaTeX-like markup.
    #
    plt.title(r'$\cal{N}(\mu, \sigma^2)$', fontsize=20)

    #
    # The file name indicates how the image will be saved and the
    # order it will appear in the movie.  If you actually wanted each
    # graph to be displayed on the screen, you would include commands
    # such as show() and draw() here.  See the matplotlib
    # documentation for details.  In this case, we are saving the
    # images directly to a file without displaying them.
    #
    filename = str('%03d' % i) + '.png'
    plt.savefig(filename, dpi=100)

    #
    # Let the user know what's happening.
    #
    print('Wrote file', filename)

    #
    # Clear the figure to make way for the next image.
    #
    plt.clf()

#
# Now that we have graphed images of the dataset, we will stitch them
# together using Mencoder to create a movie.  Each image will become
# a single frame in the movie.
#
# We want to use Python to make what would normally be a command line
# call to Mencoder.  Specifically, the command line call we want to
# emulate is (without the initial '#'):
# mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi
# See the MPlayer and Mencoder documentation for details.
#

command = ('mencoder',
           'mf://*.png',
           '-mf',
           'type=png:w=800:h=600:fps=25',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           'output.avi')

#os.spawnvp(os.P_WAIT, 'mencoder', command)

print("\n\nabout to execute:\n%s\n\n" % ' '.join(command))
subprocess.check_call(command)

print("\n\n The movie was written to 'output.avi'")

print("\n\n You may want to delete *.png now.\n\n")





********************************************
./animation/old_animation/animation_blit_gtk.py
#!/usr/bin/env python

from __future__ import print_function

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
canvas = fig.canvas

fig.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
ax.grid() # to ensure proper background restore

# create the initial line
x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x, np.sin(x), animated=True, lw=2)
canvas.draw()

# for profiling
tstart = time.time()

def update_line(*args):
    print('you are here', update_line.cnt)
    if update_line.background is None:
        update_line.background = canvas.copy_from_bbox(ax.bbox)

    # restore the clean slate background
    canvas.restore_region(update_line.background)
    # update the data
    line.set_ydata(np.sin(x+update_line.cnt/10.0))
    # just draw the animated artist
    ax.draw_artist(line)

    # just redraw the axes rectangle
    canvas.blit(ax.bbox)

    if update_line.cnt==1000:
        # print the timing info and quit
        print('FPS:' , 1000/(time.time()-tstart))
        gtk.mainquit()
        raise SystemExit

    update_line.cnt += 1
    return True

update_line.cnt = 0
update_line.background = None


def start_anim(event):
    gobject.idle_add(update_line)
    canvas.mpl_disconnect(start_anim.cid)

start_anim.cid = canvas.mpl_connect('draw_event', start_anim)



plt.show()




********************************************
./animation/old_animation/animation_blit_qt4.py
# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

from __future__ import print_function

import os
import sys

#import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE

if use_pyside:
    from PySide import QtCore, QtGui
else:
    from PyQt4 import QtCore, QtGui

ITERS = 1000

import numpy as np
import time

class BlitQT(FigureCanvas):

    def __init__(self):
        FigureCanvas.__init__(self, Figure())

        self.ax = self.figure.add_subplot(111)
        self.ax.grid()
        self.draw()

        self.old_size = self.ax.bbox.width, self.ax.bbox.height
        self.ax_background = self.copy_from_bbox(self.ax.bbox)
        self.cnt = 0

        self.x = np.arange(0,2*np.pi,0.01)
        self.sin_line, = self.ax.plot(self.x, np.sin(self.x), animated=True)
        self.cos_line, = self.ax.plot(self.x, np.cos(self.x), animated=True)
        self.draw()

        self.tstart = time.time()
        self.startTimer(10)

    def timerEvent(self, evt):
        current_size = self.ax.bbox.width, self.ax.bbox.height
        if self.old_size != current_size:
            self.old_size = current_size
            self.ax.clear()
            self.ax.grid()
            self.draw()
            self.ax_background = self.copy_from_bbox(self.ax.bbox)

        self.restore_region(self.ax_background)

        # update the data
        self.sin_line.set_ydata(np.sin(self.x+self.cnt/10.0))
        self.cos_line.set_ydata(np.cos(self.x+self.cnt/10.0))
        # just draw the animated artist
        self.ax.draw_artist(self.sin_line)
        self.ax.draw_artist(self.cos_line)
        # just redraw the axes rectangle
        self.blit(self.ax.bbox)

        if self.cnt == 0:
            # TODO: this shouldn't be necessary, but if it is excluded the
            # canvas outside the axes is not initially painted.
            self.draw()
        if self.cnt==ITERS:
            # print the timing info and quit
            print('FPS:' , ITERS/(time.time()-self.tstart))
            sys.exit()
        else:
            self.cnt += 1

app = QtGui.QApplication(sys.argv)
widget = BlitQT()
widget.show()

sys.exit(app.exec_())




********************************************
./animation/old_animation/strip_chart_demo.py
"""
Emulate an oscilloscope.  Requires the animation API introduced in
matplotlib 0.84.  See
http://www.scipy.org/wikis/topical_software/Animations for an
explanation.

This example uses gtk but does not depend on it intimately.  It just
uses the idle handler to trigger events.  You can plug this into a
different GUI that supports animation (GTKAgg, TkAgg, WXAgg) and use
your toolkits idle/timer functions.
"""
import gobject
import matplotlib
matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Scope:
    def __init__(self, ax, maxt=10, dt=0.01):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata, animated=True)
        self.ax.add_line(self.line)
        self.background = None
        self.canvas.mpl_connect('draw_event', self.update_background)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update_background(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def emitter(self, p=0.01):
        'return a random value with probability p, else 0'
        v = np.random.rand(1)
        if v>p: return 0.
        else: return np.random.rand(1)

    def update(self, *args):
        if self.background is None: return True
        y = self.emitter()
        lastt = self.tdata[-1]
        if lastt>self.tdata[0]+self.maxt: # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0]+self.maxt)
            self.ax.figure.canvas.draw()

        self.canvas.restore_region(self.background)

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        self.ax.draw_artist(self.line)

        self.canvas.blit(self.ax.bbox)
        return True


fig, ax = plt.subplots()
scope = Scope(ax)
gobject.idle_add(scope.update)

plt.show()




********************************************
./animation/old_animation/animation_blit_tk.py
# For detailed comments on animation and the techniqes used here, see
# the wiki entry http://www.scipy.org/Cookbook/Matplotlib/Animations

from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')

import sys
import matplotlib.pyplot as plt
import numpy as npy
import time

fig, ax = plt.subplots()
canvas = fig.canvas


# create the initial line
x = npy.arange(0,2*npy.pi,0.01)
line, = plt.plot(x, npy.sin(x), animated=True, lw=2)

def run(*args):
    background = canvas.copy_from_bbox(ax.bbox)
    # for profiling
    tstart = time.time()

    while 1:
        # restore the clean slate background
        canvas.restore_region(background)
        # update the data
        line.set_ydata(npy.sin(x+run.cnt/10.0))
        # just draw the animated artist
        ax.draw_artist(line)
        # just redraw the axes rectangle
        canvas.blit(ax.bbox)

        if run.cnt==1000:
            # print the timing info and quit
            print('FPS:', 1000/(time.time()-tstart))
            sys.exit()

        run.cnt += 1
run.cnt = 0


plt.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
plt.grid() # to ensure proper background restore
manager = plt.get_current_fig_manager()
manager.window.after(100, run)

plt.show()







********************************************
./animation/old_animation/gtk_timeout.py
import gobject
import numpy as np
import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0, 1)

def update():
    line.set_ydata(np.random.rand(10))
    fig.canvas.draw_idle()
    return True  # return False to terminate the updates

gobject.timeout_add(100, update)  # you can also use idle_add to update when gtk is idle
plt.show()




********************************************
./animation/old_animation/animation_blit_gtk2.py
#!/usr/bin/env python

from __future__ import print_function

"""
This example utlizes restore_region with optional bbox and xy
arguments.  The plot is continuously shifted to the left. Instead of
drawing everything again, the plot is saved (copy_from_bbox) and
restored with offset by the amount of the shift. And only newly
exposed area is drawn. This technique may reduce drawing time for some cases.
"""

import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt

class UpdateLine(object):
    def get_bg_bbox(self):
        
        return self.ax.bbox.padded(-3)
    
    def __init__(self, canvas, ax):
        self.cnt = 0
        self.canvas = canvas
        self.ax = ax

        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_pixel_offset = 0.
        

        self.x0 = 0
        self.phases = np.random.random_sample((20,)) * np.pi * 2
        self.line, = ax.plot([], [], "-", animated=True, lw=2)

        self.point, = ax.plot([], [], "ro", animated=True, lw=2)

        self.ax.set_ylim(-1.1, 1.1)

        self.background1 = None

        cmap = plt.cm.jet
        from itertools import cycle
        self.color_cycle = cycle(cmap(np.arange(cmap.N)))


    def save_bg(self):
        self.background1 = self.canvas.copy_from_bbox(self.ax.get_figure().bbox)

        self.background2 = self.canvas.copy_from_bbox(self.get_bg_bbox())


    def get_dx_data(self, dx_pixel):
        tp = self.ax.transData.inverted().transform_point
        x0, y0 = tp((0, 0))
        x1, y1 = tp((dx_pixel, 0))
        return (x1-x0)


    def restore_background_shifted(self, dx_pixel):
        """
        restore bacground shifted by dx in data coordinate. This only
        works if the data coordinate system is linear.
        """

        # restore the clean slate background
        self.canvas.restore_region(self.background1)

        # restore subregion (x1+dx, y1, x2, y2) of the second bg 
        # in a offset position (x1-dx, y1)
        x1, y1, x2, y2 = self.background2.get_extents()
        self.canvas.restore_region(self.background2,
                                   bbox=(x1+dx_pixel, y1, x2, y2),
                                   xy=(x1-dx_pixel, y1))

        return dx_pixel

    def on_draw(self, *args):
        self.save_bg()
        return False
    
    def update_line(self, *args):

        if self.background1 is None:
            return True
        
        cur_time = time.time()
        pixel_offset = int((cur_time - self.start_time)*100.)
        dx_pixel = pixel_offset - self.prev_pixel_offset
        self.prev_pixel_offset = pixel_offset
        dx_data = self.get_dx_data(dx_pixel) #cur_time - self.prev_time)
        
        x0 = self.x0
        self.x0 += dx_data
        self.prev_time = cur_time

        self.ax.set_xlim(self.x0-2, self.x0+0.1)


        # restore background which will plot lines from previous plots
        self.restore_background_shifted(dx_pixel) #x0, self.x0)
        # This restores lines between [x0-2, x0]



        self.line.set_color(self.color_cycle.next())

        # now plot line segment within [x0, x0+dx_data], 
        # Note that we're only plotting a line between [x0, x0+dx_data].
        xx = np.array([x0, self.x0])
        self.line.set_xdata(xx)

        # the for loop below could be improved by using collection.
        [(self.line.set_ydata(np.sin(xx+p)),
          self.ax.draw_artist(self.line)) \
         for p in self.phases]

        self.background2 = canvas.copy_from_bbox(self.get_bg_bbox())

        self.point.set_xdata([self.x0])

        [(self.point.set_ydata(np.sin([self.x0+p])),
          self.ax.draw_artist(self.point)) \
         for p in self.phases]


        self.ax.draw_artist(self.ax.xaxis)
        self.ax.draw_artist(self.ax.yaxis)

        self.canvas.blit(self.ax.get_figure().bbox)


        dt = (time.time()-tstart)
        if dt>15:
            # print the timing info and quit
            print('FPS:' , self.cnt/dt)
            gtk.main_quit()
            raise SystemExit

        self.cnt += 1
        return True


plt.rcParams["text.usetex"] = False

fig, ax = plt.subplots()
ax.xaxis.set_animated(True)
ax.yaxis.set_animated(True)
canvas = fig.canvas

fig.subplots_adjust(left=0.2, bottom=0.2)
canvas.draw()

# for profiling
tstart = time.time()

ul = UpdateLine(canvas, ax)
gobject.idle_add(ul.update_line)

canvas.mpl_connect('draw_event', ul.on_draw)

plt.show()




********************************************
./animation/old_animation/simple_idle_wx.py
"""
A simple example of an animated plot using a wx backend
"""
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.arange(0, 2*np.pi, 0.1)
line, = ax.plot(t, np.sin(t))
dt = 0.05

def update_line(idleevent):
    if update_line.i==200:
        return False
    print('animate', update_line.i)
    line.set_ydata(np.sin(t+update_line.i/10.))
    fig.canvas.draw_idle()                 # redraw the canvas
    update_line.i += 1
update_line.i = 0

import wx
wx.EVT_IDLE(wx.GetApp(), update_line)
plt.show()




********************************************
./animation/strip_chart_demo.py
"""
Emulate an oscilloscope.  Requires the animation API introduced in
matplotlib 1.0 SVN.
"""
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Scope:
    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt: # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter(p=0.03):
    'return a random value with probability p, else 0'
    while True:
        v = np.random.rand(1)
        if v > p:
            yield 0.
        else:
            yield np.random.rand(1)

fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=10,
    blit=True)


plt.show()




********************************************
./animation/double_pendulum_animated.py
# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G =  9.8 # acceleration due to gravity, in m/s^2
L1 = 1.0 # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0 # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2]-state[0]
    den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
               + M2*G*sin(state[2])*cos(del_) + M2*L2*state[3]*state[3]*sin(del_)
               - (M1+M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)
               + (M1+M2)*G*sin(state[0])*cos(del_)
               - (M1+M2)*L1*state[1]*state[1]*sin(del_)
               - (M1+M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.1 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

rad = pi/180

# initial state
state = np.array([th1, w1, th2, w2])*pi/180.

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:,0])
y1 = -L1*cos(y[:,0])

x2 = L2*sin(y[:,2]) + x1
y2 = -L2*cos(y[:,2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=25, blit=True, init_func=init)

#ani.save('double_pendulum.mp4', fps=15, clear_temp=True)
plt.show()




********************************************
./color/colormaps_reference.py
"""
Reference for colormaps included with Matplotlib.

This reference example shows all colormaps included with Matplotlib. Note that
any colormap listed here can be reversed by appending "_r" (e.g., "pink_r").
These colormaps are divided into the following categories:

Sequential:
    These colormaps are approximately monochromatic colormaps varying smoothly
    between two color tones---usually from low saturation (e.g. white) to high
    saturation (e.g. a bright blue). Sequential colormaps are ideal for
    representing most scientific data since they show a clear progression from
    low-to-high values.

Diverging:
    These colormaps have a median value (usually light in color) and vary
    smoothly to two different color tones at high and low values. Diverging
    colormaps are ideal when your data has a median value that is significant
    (e.g.  0, such that positive and negative values are represented by
    different colors of the colormap).

Qualitative:
    These colormaps vary rapidly in color. Qualitative colormaps are useful for
    choosing a set of discrete colors. For example::

        color_list = plt.cm.Set3(np.linspace(0, 1, 12))

    gives a list of RGB colors that are good for plotting a series of lines on
    a dark background.

Miscellaneous:
    Colormaps that don't fit into the categories above.

"""
import numpy as np
import matplotlib.pyplot as plt


cmaps = [('Sequential',     ['binary', 'Blues', 'BuGn', 'BuPu', 'gist_yarg',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_gray', 'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'hsv', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3', 'spectral']),
         ('Miscellaneous',  ['gist_earth', 'gist_ncar', 'gist_rainbow',
                             'gist_stern', 'jet', 'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'ocean', 'rainbow',
                             'terrain', 'flag', 'prism'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plot_color_gradients(cmap_category, cmap_list):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()

for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list)

plt.show()




********************************************
./color/color_cycle_demo.py
"""
Demo of custom color-cycle settings to control colors for multi-line plots.

This example demonstrates two different APIs:

    1. Setting the default rc-parameter specifying the color cycle.
       This affects all subsequent plots.
    2. Setting the color cycle for a specific axes. This only affects a single
       axes.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi)
offsets = np.linspace(0, 2*np.pi, 4, endpoint=False)
# Create array with shifted-sine curve along each column
yy = np.transpose([np.sin(x + phi) for phi in offsets])

plt.rc('lines', linewidth=4)
fig, (ax0, ax1)  = plt.subplots(nrows=2)

plt.rc('axes', color_cycle=['r', 'g', 'b', 'y'])
ax0.plot(yy)
ax0.set_title('Set default color cycle to rgby')

ax1.set_color_cycle(['c', 'm', 'y', 'k'])
ax1.plot(yy)
ax1.set_title('Set axes color cycle to cmyk')

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.3)
plt.show()






********************************************
./pylab_examples/image_origin.py
#!/usr/bin/env python
"""
You can specify whether images should be plotted with the array origin
x[0,0] in the upper left or upper right by using the origin parameter.
You can also control the default be setting image.origin in your
matplotlibrc file; see http://matplotlib.org/matplotlibrc
"""
from pylab import *

x = arange(100.0); x.shape = 10,10

interp = 'bilinear';
#interp = 'nearest';
lim = -2,11,-2,6
subplot(211, axisbg='g')
title('blue should be up')
imshow(x, origin='upper', interpolation=interp)
#axis(lim)

subplot(212, axisbg='y')
title('blue should be down')
imshow(x, origin='lower', interpolation=interp)
#axis(lim)
show()




********************************************
./pylab_examples/demo_text_rotation_mode.py

#clf()
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

def test_rotation_mode(fig, mode, subplot_location):
    ha_list = "left center right".split()
    va_list = "top center baseline bottom".split()
    grid = ImageGrid(fig, subplot_location,
                    nrows_ncols=(len(va_list), len(ha_list)),
                    share_all=True, aspect=True, #label_mode='1',
                    cbar_mode=None)

    for ha, ax in zip(ha_list, grid.axes_row[-1]):
        ax.axis["bottom"].label.set_text(ha)

    grid.axes_row[0][1].set_title(mode, size="large")

    for va, ax in zip(va_list, grid.axes_column[0]):
        ax.axis["left"].label.set_text(va)

    i = 0
    for va in va_list:
        for ha in ha_list:
            ax = grid[i]
            for axis in ax.axis.values():
                axis.toggle(ticks=False, ticklabels=False)

            ax.text(0.5, 0.5, "Tpg",
                    size="large", rotation=40,
                    bbox=dict(boxstyle="square,pad=0.",
                              ec="none", fc="0.5", alpha=0.5),
                    ha=ha, va=va,
                    rotation_mode=mode)
            ax.axvline(0.5)
            ax.axhline(0.5)
            i += 1

if 1:
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(5.5,4 ))
    fig.clf()

    test_rotation_mode(fig, "default", 121)
    test_rotation_mode(fig, "anchor", 122)
    plt.show()




********************************************
./pylab_examples/image_masked.py
#!/usr/bin/env python
'''imshow with masked array input and out-of-range colors.

    The second subplot illustrates the use of BoundaryNorm to
    get a filled contour effect.
'''

from pylab import *
from numpy import ma
import matplotlib.colors as colors

delta = 0.025
x = y = arange(-3.0, 3.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z2-Z1)  # difference of Gaussians

# Set up a colormap:
palette = cm.gray
palette.set_over('r', 1.0)
palette.set_under('g', 1.0)
palette.set_bad('b', 1.0)
# Alternatively, we could use
# palette.set_bad(alpha = 0.0)
# to make the bad region transparent.  This is the default.
# If you comment out all the palette.set* lines, you will see
# all the defaults; under and over will be colored with the
# first and last colors in the palette, respectively.
Zm = ma.masked_where(Z > 1.2, Z)

# By setting vmin and vmax in the norm, we establish the
# range to which the regular palette color scale is applied.
# Anything above that range is colored based on palette.set_over, etc.

subplot(1,2,1)
im = imshow(Zm, interpolation='bilinear',
    cmap=palette,
    norm = colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False),
    origin='lower', extent=[-3,3,-3,3])
title('Green=low, Red=high, Blue=bad')
colorbar(im, extend='both', orientation='horizontal', shrink=0.8)

subplot(1,2,2)
im = imshow(Zm, interpolation='nearest',
    cmap=palette,
    norm = colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                        ncolors=256, clip = False),
    origin='lower', extent=[-3,3,-3,3])
title('With BoundaryNorm')
colorbar(im, extend='both', spacing='proportional',
                orientation='horizontal', shrink=0.8)

show()





********************************************
./pylab_examples/subplots_adjust.py
from pylab import *


subplot(211)
imshow(rand(100,100), cmap=cm.BuPu_r)
subplot(212)
imshow(rand(100,100), cmap=cm.BuPu_r)

subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = axes([0.85, 0.1, 0.075, 0.8])
colorbar(cax=cax)
show()




********************************************
./pylab_examples/legend_demo3.py
#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['legend.fancybox'] = True
import matplotlib.pyplot as plt
import numpy as np

def myplot(ax):
    t1 = np.arange(0.0, 1.0, 0.1)
    for n in [1, 2, 3, 4]:
        ax.plot(t1, t1**n, label="n=%d"%(n,))

ax1 = plt.subplot(3,1,1)
ax1.plot([1], label="multi\nline")
ax1.plot([1], label="$2^{2^2}$")
ax1.plot([1], label=r"$\frac{1}{2}\pi$")
ax1.legend(loc=1, ncol=3, shadow=True)

ax2 = plt.subplot(3,1,2)
myplot(ax2)
ax2.legend(loc="center left", bbox_to_anchor=[0.5, 0.5],
           ncol=2, shadow=True, title="Legend")
ax2.get_legend().get_title().set_color("red")

ax3 = plt.subplot(3,1,3)
myplot(ax3)
ax3.legend(shadow=True, fancybox=True)


plt.draw()
plt.show()







********************************************
./pylab_examples/dashpointlabel.py
import matplotlib.pyplot as plt

DATA = ((1, 3),
        (2, 4),
        (3, 1),
        (4, 2))
# dash_style =
#     direction, length, (text)rotation, dashrotation, push
# (The parameters are varied to show their effects,
# not for visual appeal).
dash_style = (
    (0, 20, -15, 30, 10),
    (1, 30, 0, 15, 10),
    (0, 40, 15, 15, 10),
    (1, 20, 30, 60, 10),
    )

fig, ax = plt.subplots()

(x,y) = zip(*DATA)
ax.plot(x, y, marker='o')
for i in range(len(DATA)):
    (x,y) = DATA[i]
    (dd, dl, r, dr, dp) = dash_style[i]
    #print 'dashlen call', dl
    t = ax.text(x, y, str((x,y)), withdash=True,
               dashdirection=dd,
               dashlength=dl,
               rotation=r,
               dashrotation=dr,
               dashpush=dp,
               )

ax.set_xlim((0.0, 5.0))
ax.set_ylim((0.0, 5.0))

plt.show()





********************************************
./pylab_examples/axes_zoom_effect.py
from matplotlib.transforms import Bbox, TransformedBbox, \
     blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
     BboxConnectorPatch


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           #loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    ax1 : the main axes
    ax1 : the zoomed axes
    (xmin,xmax) : the limits of the colored area in both plot axes.

    connect ax1 & ax2. The x-range of (xmin, xmax) in both axes will
    be marked.  The keywords parameters will be used ti create
    patches.

    """

    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox, trans2)

    prop_patches=kwargs.copy()
    prop_patches["ec"]="none"
    prop_patches["alpha"]=0.2

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                     prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect02(ax1, ax2, **kwargs):
    """
    ax1 : the main axes
    ax1 : the zoomed axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches=kwargs.copy()
    prop_patches["ec"]="none"
    prop_patches["alpha"]=0.2

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                     prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


import matplotlib.pyplot as plt

plt.figure(1, figsize=(5,5))
ax1 = plt.subplot(221)
ax2 = plt.subplot(212)
ax2.set_xlim(0, 1)
ax2.set_xlim(0, 5)
zoom_effect01(ax1, ax2, 0.2, 0.8)


ax1 = plt.subplot(222)
ax1.set_xlim(2, 3)
ax2.set_xlim(0, 5)
zoom_effect02(ax1, ax2)

plt.show()




********************************************
./pylab_examples/scatter_profile.py
#!/usr/bin/env python
# -*- noplot -*-

"""
N       Classic     Base renderer    Ext renderer
20       0.22           0.14            0.14
100      0.16           0.14            0.13
1000     0.45           0.26            0.17
10000    3.30           1.31            0.53
50000    19.30          6.53            1.98
"""
from __future__ import print_function
import pylab

import time


for N in (20,100,1000,10000,50000):
    tstart = time.time()
    x = 0.9*pylab.rand(N)
    y = 0.9*pylab.rand(N)
    s = 20*pylab.rand(N)
    pylab.scatter(x,y,s)
    print ('%d symbols in %1.2f s' % (N, time.time()-tstart))




********************************************
./pylab_examples/fonts_demo.py
#!/usr/bin/env python
"""
Show how to set custom font properties.

For interactive users, you can also use kwargs to the text command,
which requires less typing.  See examples/fonts_demo_kw.py
"""
from matplotlib.font_manager import FontProperties
from pylab import *

subplot(111, axisbg='w')

font0 = FontProperties()
alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
###  Show family options

family = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

font1 = font0.copy()
font1.set_size('large')

t = text(-0.8, 0.9, 'family', fontproperties=font1,
         **alignment)

yp = [0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5]

for k in range(5):
    font = font0.copy()
    font.set_family(family[k])
    if k == 2:
        font.set_name('Script MT')
    t = text(-0.8, yp[k], family[k], fontproperties=font,
             **alignment)

###  Show style options

style  = ['normal', 'italic', 'oblique']

t = text(-0.4, 0.9, 'style', fontproperties=font1,
         **alignment)

for k in range(3):
    font = font0.copy()
    font.set_family('sans-serif')
    font.set_style(style[k])
    t = text(-0.4, yp[k], style[k], fontproperties=font,
             **alignment)

###  Show variant options

variant= ['normal', 'small-caps']

t = text(0.0, 0.9, 'variant', fontproperties=font1,
         **alignment)

for k in range(2):
    font = font0.copy()
    font.set_family('serif')
    font.set_variant(variant[k])
    t = text( 0.0, yp[k], variant[k], fontproperties=font,
             **alignment)

###  Show weight options

weight = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = text( 0.4, 0.9, 'weight', fontproperties=font1,
         **alignment)

for k in range(7):
    font = font0.copy()
    font.set_weight(weight[k])
    t = text( 0.4, yp[k], weight[k], fontproperties=font,
             **alignment)

###  Show size options

size  = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = text( 0.8, 0.9, 'size', fontproperties=font1,
         **alignment)

for k in range(7):
    font = font0.copy()
    font.set_size(size[k])
    t = text( 0.8, yp[k], size[k], fontproperties=font,
             **alignment)

###  Show bold italic

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-small')
t = text(0, 0.1, 'bold italic', fontproperties=font,
         **alignment)

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('medium')
t = text(0, 0.2, 'bold italic', fontproperties=font,
         **alignment)

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-large')
t = text(0, 0.3, 'bold italic', fontproperties=font,
         **alignment)

axis([-1,1,0,1])

show()




********************************************
./pylab_examples/customize_rc.py

"""
I'm not trying to make a good looking figure here, but just to show
some examples of customizing rc params on the fly

If you like to work interactively, and need to create different sets
of defaults for figures (eg one set of defaults for publication, one
set for interactive exploration), you may want to define some
functions in a custom module that set the defaults, eg

def set_pub():
    rc('font', weight='bold')    # bold fonts are easier to see
    rc('tick', labelsize=15)     # tick labels bigger
    rc('lines', lw=1, color='k') # thicker black lines (no budget for color!)
    rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rc('savefig', dpi=300)       # higher res outputs



Then as you are working interactively, you just need to do

>>> set_pub()
>>> subplot(111)
>>> plot([1,2,3])
>>> savefig('myfig')
>>> rcdefaults()  # restore the defaults

"""
from pylab import *

subplot(311)
plot([1,2,3])

# the axes attributes need to be set before the call to subplot
rc('font', weight='bold')
rc('xtick.major', size=5, pad=7)
rc('xtick', labelsize=15)

# using aliases for color, linestyle and linewidth; gray, solid, thick
rc('grid', c='0.5', ls='-', lw=5)
rc('lines', lw=2, color='g')
subplot(312)

plot([1,2,3])
grid(True)

rcdefaults()
subplot(313)
plot([1,2,3])
grid(True)
show()




********************************************
./pylab_examples/legend_translucent.py
#!/usr/bin/python
#
# Show how to add a translucent legend

# import pyplot module
import matplotlib.pyplot as plt

# draw 2 crossing lines
plt.plot([0,1], label='going up')
plt.plot([1,0], label='going down')

# add the legend in the middle of the plot
leg = plt.legend(fancybox=True, loc='center')
# set the alpha value of the legend: it will be translucent
leg.get_frame().set_alpha(0.5)

# show the plot
plt.show()




********************************************
./pylab_examples/pie_demo2.py
"""
Make a pie charts of varying size - see
http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pie for the docstring.

This example shows a basic pie charts with labels optional features,
like autolabeling the percentage, offsetting a slice with "explode"
and adding a shadow, in different sizes.

"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Some data

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]

explode=(0, 0.05, 0, 0)

# Make square figures and axes

the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0], aspect=1)

plt.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)

plt.subplot(the_grid[0, 1], aspect=1)

plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)

plt.subplot(the_grid[1, 0], aspect=1)

patches, texts, autotexts = plt.pie(fracs, labels=labels,
                                    autopct='%.0f%%',
                                    shadow=True, radius=0.5)

# Make the labels on the small plot easier to read.
for t in texts:
    t.set_size('smaller')
for t in autotexts:
    t.set_size('x-small')
autotexts[0].set_color('y')

plt.subplot(the_grid[1, 1], aspect=1)

patches, texts, autotexts = plt.pie(fracs, explode=explode,
                                labels=labels, autopct='%.0f%%',
                                shadow=False, radius=0.5)
                                # Turn off shadow for tiny plot
                                # with exploded slice.
for t in texts:
    t.set_size('smaller')
for t in autotexts:
    t.set_size('x-small')
autotexts[0].set_color('y')

plt.show()




********************************************
./pylab_examples/legend_scatter.py
#!/usr/bin/env python
from pylab import *

N=1000

props = dict( alpha=0.5, edgecolors='none' )

handles = []
colours = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
colours = ['red', 'green', 'blue']
for colour in colours:
    x, y = rand(2,N)
    s = 400.0 * rand(N)
    handles.append(scatter(x, y, c=colour, s=s, **props))

legend(handles, colours)
grid(True)

show()






********************************************
./pylab_examples/symlog_demo.py
#!/usr/bin/env python
from pylab import *

dt = 0.01
x = arange(-50.0, 50.0, dt)
y = arange(0, 100.0, dt)

subplot(311)
plot(x, y)
xscale('symlog')
ylabel('symlogx')
grid(True)
gca().xaxis.grid(True, which='minor')  # minor grid on too

subplot(312)
plot(y, x)
yscale('symlog')
ylabel('symlogy')


subplot(313)
plot(x, np.sin(x / 3.0))
xscale('symlog')
yscale('symlog', linthreshy=0.015)
grid(True)
ylabel('symlog both')

show()




********************************************
./pylab_examples/demo_ribbon_box.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import BboxImage

from matplotlib._png import read_png
import matplotlib.colors
from matplotlib.cbook import get_sample_data

class RibbonBox(object):

    original_image = read_png(get_sample_data("Minduka_Present_Blue_Pack.png",
                                              asfileobj=False))
    cut_location = 70
    b_and_h = original_image[:,:,2]
    color = original_image[:,:,2] - original_image[:,:,0]
    alpha = original_image[:,:,3]
    nx = original_image.shape[1]

    def __init__(self, color):
        rgb = matplotlib.colors.colorConverter.to_rgb(color)

        im = np.empty(self.original_image.shape,
                      self.original_image.dtype)


        im[:,:,:3] = self.b_and_h[:,:,np.newaxis]
        im[:,:,:3] -= self.color[:,:,np.newaxis]*(1.-np.array(rgb))
        im[:,:,3] = self.alpha

        self.im = im


    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)

        stretched_image = np.empty((ny2, nx, nch),
                                   self.im.dtype)
        cut = self.im[self.cut_location,:,:]
        stretched_image[:,:,:] = cut
        stretched_image[:self.cut_location,:,:] = \
                self.im[:self.cut_location,:,:]
        stretched_image[-(ny-self.cut_location):,:,:] = \
                self.im[-(ny-self.cut_location):,:,:]

        self._cached_im = stretched_image
        return stretched_image



class RibbonBoxImage(BboxImage):
    zorder = 1

    def __init__(self, bbox, color,
                 cmap = None,
                 norm = None,
                 interpolation=None,
                 origin=None,
                 filternorm=1,
                 filterrad=4.0,
                 resample = False,
                 **kwargs
                 ):

        BboxImage.__init__(self, bbox,
                           cmap = cmap,
                           norm = norm,
                           interpolation=interpolation,
                           origin=origin,
                           filternorm=filternorm,
                           filterrad=filterrad,
                           resample = resample,
                           **kwargs
                           )

        self._ribbonbox = RibbonBox(color)
        self._cached_ny = None


    def draw(self, renderer, *args, **kwargs):

        bbox = self.get_window_extent(renderer)
        stretch_factor = bbox.height / bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self._cached_ny != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)
            self._cached_ny = ny

        BboxImage.draw(self, renderer, *args, **kwargs)


if 1:
    from matplotlib.transforms import Bbox, TransformedBbox
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots()

    years = np.arange(2004, 2009)
    box_colors = [(0.8, 0.2, 0.2),
                  (0.2, 0.8, 0.2),
                  (0.2, 0.2, 0.8),
                  (0.7, 0.5, 0.8),
                  (0.3, 0.8, 0.7),
                  ]
    heights = np.random.random(years.shape) * 7000 + 3000

    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)

    for year, h, bc in zip(years, heights, box_colors):
        bbox0 = Bbox.from_extents(year-0.4, 0., year+0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        rb_patch = RibbonBoxImage(bbox, bc, interpolation="bicubic")

        ax.add_artist(rb_patch)

        ax.annotate(r"%d" % (int(h/100.)*100),
                    (year, h), va="bottom", ha="center")

    patch_gradient = BboxImage(ax.bbox,
                               interpolation="bicubic",
                               zorder=0.1,
                               )
    gradient = np.zeros((2, 2, 4), dtype=np.float)
    gradient[:,:,:3] = [1, 1, 0.]
    gradient[:,:,3] = [[0.1, 0.3],[0.3, 0.5]] # alpha channel
    patch_gradient.set_array(gradient)
    ax.add_artist(patch_gradient)


    ax.set_xlim(years[0]-0.5, years[-1]+0.5)
    ax.set_ylim(0, 10000)

    fig.savefig('ribbon_box.png')
    plt.show()





********************************************
./pylab_examples/stix_fonts_demo.py
#!/usr/bin/env python

from __future__ import unicode_literals

import os, sys, re

import gc

stests = [
    r'$\mathcircled{123} \mathrm{\mathcircled{123}} \mathbf{\mathcircled{123}}$',
    r'$\mathsf{Sans \Omega} \mathrm{\mathsf{Sans \Omega}} \mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard \pi}$',
    r'$\mathrm{\mathbb{Blackboard \pi}}$',
    r'$\mathbf{\mathbb{Blackboard \pi}}$',
    r'$\mathfrak{Fraktur} \mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$']

if sys.maxunicode > 0xffff:
    s = r'Direct Unicode: $\u23ce \mathrm{\ue0f2 \U0001D538}$'

from pylab import *

def doall():
    tests = stests

    figure(figsize=(8, (len(tests) * 1) + 2))
    plot([0, 0], 'r')
    grid(False)
    axis([0, 3, -len(tests), 0])
    yticks(arange(len(tests)) * -1)
    for i, s in enumerate(tests):
        #print (i, s.encode("ascii", "backslashreplace"))
        text(0.1, -i, s, fontsize=32)

    savefig('stix_fonts_example')
    #close('all')
    show()

if '--latex' in sys.argv:
    fd = open("stix_fonts_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i, s in enumerate(stests):
        s = re.sub(r"(?<!\\)\$", "$$", s)
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex stix_fonts_examples.ltx")
else:
    doall()




********************************************
./pylab_examples/table_demo.py
"""
Demo of table function to display a table within a plot.
"""
import numpy as np
import matplotlib.pyplot as plt


data = [[  66386,  174296,   75131,  577908,   32015],
        [  58230,  381139,   78045,   99308,  160454],
        [  89135,   80552,  152558,  497981,  603535],
        [  78415,   81858,  150656,  193263,   69638],
        [ 139361,  331509,  343164,  781380,   52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(columns)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.array([0.0] * len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x/1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()




********************************************
./pylab_examples/mathtext_examples.py
"""
Selected features of Matplotlib's math rendering engine.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import sys
import re
import gc

# Selection of features following "Writing mathematical expressions" tutorial
mathtext_titles = {
    0: "Header demo",
    1: "Subscripts and superscripts",
    2: "Fractions, binomials and stacked numbers",
    3: "Radicals",
    4: "Fonts",
    5: "Accents",
    6: "Greek, Hebrew",
    7: "Delimiters, functions and Symbols"}
n_lines = len(mathtext_titles)

# Randomly picked examples
mathext_demos = {
    0: r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = "
    r"U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} "
    r"\int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ "
    r"U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_"
    r"{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",

    1: r"$\alpha_i > \beta_i,\ "
    r"\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau},\ "
    r"\ldots$",

    2: r"$\frac{3}{4},\ \binom{3}{4},\ \stackrel{3}{4},\ "
    r"\left(\frac{5 - \frac{1}{x}}{4}\right),\ \ldots$",

    3: r"$\sqrt{2},\ \sqrt[3]{x},\ \ldots$",

    4: r"$\mathrm{Roman}\ , \ \mathit{Italic}\ , \ \mathtt{Typewriter} \ "
    r"\mathrm{or}\ \mathcal{CALLIGRAPHY}$",

    5: r"$\acute a,\ \bar a,\ \breve a,\ \dot a,\ \ddot a, \ \grave a, \ "
    r"\hat a,\ \tilde a,\ \vec a,\ \widehat{xyz},\ \widetilde{xyz},\ "
    r"\ldots$",

    6: r"$\alpha,\ \beta,\ \chi,\ \delta,\ \lambda,\ \mu,\ "
    r"\Delta,\ \Gamma,\ \Omega,\ \Phi,\ \Pi,\ \Upsilon,\ \nabla,\ "
    r"\aleph,\ \beth,\ \daleth,\ \gimel,\ \ldots$",

    7: r"$\coprod,\ \int,\ \oint,\ \prod,\ \sum,\ "
    r"\log,\ \sin,\ \approx,\ \oplus,\ \star,\ \varpropto,\ "
    r"\infty,\ \partial,\ \Re,\ \leftrightsquigarrow, \ \ldots$"}


def doall():
    # Colors used in mpl online documentation.
    mpl_blue_rvb = (191./255., 209./256., 212./255.)
    mpl_orange_rvb = (202/255., 121/256., 0./255.)
    mpl_grey_rvb = (51./255., 51./255., 51./255.)

    # Creating figure and axis.
    plt.figure(figsize=(6, 7))
    plt.axes([0.01, 0.01, 0.98, 0.90], axisbg="white", frameon=True)
    plt.gca().set_xlim(0., 1.)
    plt.gca().set_ylim(0., 1.)
    plt.gca().set_title("Matplotlib's math rendering engine",
                        color=mpl_grey_rvb, fontsize=14, weight='bold')
    plt.gca().set_xticklabels("", visible=False)
    plt.gca().set_yticklabels("", visible=False)

    # Gap between lines in axes coords
    line_axesfrac = (1. / (n_lines))

    # Plotting header demonstration formula
    full_demo = mathext_demos[0]
    plt.annotate(full_demo,
                 xy=(0.5, 1. - 0.59*line_axesfrac),
                 xycoords='data', color=mpl_orange_rvb, ha='center',
                 fontsize=20)

    # Plotting features demonstration formulae
    for i_line in range(1, n_lines):
        baseline = 1. - (i_line)*line_axesfrac
        baseline_next = baseline - line_axesfrac*1.
        title = mathtext_titles[i_line] + ":"
        fill_color = ['white', mpl_blue_rvb][i_line % 2]
        plt.fill_between([0., 1.], [baseline, baseline],
                         [baseline_next, baseline_next],
                         color=fill_color, alpha=0.5)
        plt.annotate(title,
                     xy=(0.07, baseline - 0.3*line_axesfrac),
                     xycoords='data', color=mpl_grey_rvb, weight='bold')
        demo = mathext_demos[i_line]
        plt.annotate(demo,
                     xy=(0.05, baseline - 0.75*line_axesfrac),
                     xycoords='data', color=mpl_grey_rvb,
                     fontsize=16)

    for i in range(n_lines):
        s = mathext_demos[i]
        print(i, s)
    plt.show()

if '--latex' in sys.argv:
    # Run: python mathtext_examples.py --latex
    # Need amsmath and amssymb packages.
    fd = open("mathtext_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\usepackage{amsmath, amssymb}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i in range(n_lines):
        s = mathext_demos[i]
        s = re.sub(r"(?<!\\)\$", "$$", s)
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex mathtext_examples.ltx")
else:
    doall()




********************************************
./pylab_examples/cohere_demo.py
#!/usr/bin/env python
"""
Compute the coherence of two signals
"""
import numpy as np
import matplotlib.pyplot as plt

# make a little extra space between the subplots
plt.subplots_adjust(wspace=0.5)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2
r = np.exp(-t/0.05)

cnse1 = np.convolve(nse1, r, mode='same')*dt   # colored noise 1
cnse2 = np.convolve(nse2, r, mode='same')*dt   # colored noise 2

# two signals with a coherent part and a random part
s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1
s2 = 0.01*np.sin(2*np.pi*10*t) + cnse2

plt.subplot(211)
plt.plot(t, s1, 'b-', t, s2, 'g-')
plt.xlim(0,5)
plt.xlabel('time')
plt.ylabel('s1 and s2')
plt.grid(True)

plt.subplot(212)
cxy, f = plt.cohere(s1, s2, 256, 1./dt)
plt.ylabel('coherence')
plt.show()






********************************************
./pylab_examples/loadrec.py
from __future__ import print_function
from matplotlib import mlab
from pylab import figure, show
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)
a = mlab.csv2rec(datafile)
a.sort()
print(a.dtype)

fig = figure()
ax = fig.add_subplot(111)
ax.plot(a.date, a.adj_close, '-')
fig.autofmt_xdate()

# if you have xlwt installed, you can output excel
try:
    import mpl_toolkits.exceltools as exceltools
    exceltools.rec2excel(a, 'test.xls')
except ImportError:
    pass
show()




********************************************
./pylab_examples/fancybox_demo.py
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch


# Bbox object around which the fancy box will be drawn.
bb = mtransforms.Bbox([[0.3, 0.4], [0.7, 0.6]])

def draw_bbox(ax, bb):
    # boxstyle=square with pad=0, i.e. bbox itself.
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height), 
                            boxstyle="square,pad=0.",
                            ec="k", fc="none", zorder=10.,
                            )
    ax.add_patch(p_bbox)

def test1(ax):

    # a fancy box with round corners. pad=0.1
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.1",
                             fc=(1., .8, 1.),
                             ec=(1., 0.5, 1.))


    ax.add_patch(p_fancy)

    ax.text(0.1, 0.8,
            r' boxstyle="round,pad=0.1"',
            size=10, transform=ax.transAxes)

    # draws control points for the fancy box.
    #l = p_fancy.get_path().vertices
    #ax.plot(l[:,0], l[:,1], ".")

    # draw the original bbox in black
    draw_bbox(ax, bb)


def test2(ax):

    # bbox=round has two optional argument. pad and rounding_size.
    # They can be set during the initialization.
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.1",
                             fc=(1., .8, 1.),
                             ec=(1., 0.5, 1.))


    ax.add_patch(p_fancy)

    # boxstyle and its argument can be later modified with
    # set_boxstyle method. Note that the old attributes are simply
    # forgotten even if the boxstyle name is same.

    p_fancy.set_boxstyle("round,pad=0.1, rounding_size=0.2")
    #or
    #p_fancy.set_boxstyle("round", pad=0.1, rounding_size=0.2)

    ax.text(0.1, 0.8,
            ' boxstyle="round,pad=0.1\n rounding\\_size=0.2"',
            size=10, transform=ax.transAxes)

    # draws control points for the fancy box.
    #l = p_fancy.get_path().vertices
    #ax.plot(l[:,0], l[:,1], ".")
    
    draw_bbox(ax, bb)



def test3(ax):

    # mutation_scale determine overall scale of the mutation,
    # i.e. both pad and rounding_size is scaled according to this
    # value.
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.1",
                             mutation_scale=2.,
                             fc=(1., .8, 1.),
                             ec=(1., 0.5, 1.))


    ax.add_patch(p_fancy)

    ax.text(0.1, 0.8,
            ' boxstyle="round,pad=0.1"\n mutation\\_scale=2',
            size=10, transform=ax.transAxes)

    # draws control points for the fancy box.
    #l = p_fancy.get_path().vertices
    #ax.plot(l[:,0], l[:,1], ".")
    
    draw_bbox(ax, bb)



def test4(ax):

    # When the aspect ratio of the axes is not 1, the fancy box may
    # not be what you expected (green)
    
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.2",
                             fc="none",
                             ec=(0., .5, 0.), zorder=4)


    ax.add_patch(p_fancy)


    # You can compensate this by setting the mutation_aspect (pink).
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.3",
                             mutation_aspect=.5, 
                             fc=(1., 0.8, 1.),
                             ec=(1., 0.5, 1.))


    ax.add_patch(p_fancy)

    ax.text(0.1, 0.8,
            ' boxstyle="round,pad=0.3"\n mutation\\_aspect=.5',
            size=10, transform=ax.transAxes)
    
    draw_bbox(ax, bb)



def test_all():
    plt.clf()

    ax = plt.subplot(2, 2, 1)
    test1(ax)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_title("test1")
    ax.set_aspect(1.)


    ax = plt.subplot(2, 2, 2)
    ax.set_title("test2")
    test2(ax)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_aspect(1.)

    ax = plt.subplot(2, 2, 3)
    ax.set_title("test3")
    test3(ax)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_aspect(1)

    ax = plt.subplot(2, 2, 4)
    ax.set_title("test4")
    test4(ax)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0., 1.)
    ax.set_aspect(2.)

    plt.draw()
    plt.show()

test_all()




********************************************
./pylab_examples/custom_cmap.py
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

"""

Example: suppose you want red to increase from 0 to 1 over the bottom
half, green to do the same over the middle half, and blue over the top
half.  Then you would use:

cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 1.0))}

If, as in this example, there are no discontinuities in the r, g, and b
components, then it is quite simple: the second and third element of
each tuple, above, is the same--call it "y".  The first element ("x")
defines interpolation intervals over the full range of 0 to 1, and it
must span that whole range.  In other words, the values of x divide the
0-to-1 range into a set of segments, and y gives the end-point color
values for each segment.

Now consider the green. cdict['green'] is saying that for
0 <= x <= 0.25, y is zero; no green.
0.25 < x <= 0.75, y varies linearly from 0 to 1.
x > 0.75, y remains at 1, full green.

If there are discontinuities, then it is a little more complicated.
Label the 3 elements in each row in the cdict entry for a given color as
(x, y0, y1).  Then for values of x between x[i] and x[i+1] the color
value is interpolated between y1[i] and y0[i+1].

Going back to the cookbook example, look at cdict['red']; because y0 !=
y1, it is saying that for x from 0 to 0.5, red increases from 0 to 1,
but then it jumps down, so that for x from 0.5 to 1, red increases from
0.7 to 1.  Green ramps from 0 to 1 as x goes from 0 to 0.5, then jumps
back to 0, and ramps back to 1 as x goes from 0.5 to 1.

row i:   x  y0  y1
                /
               /
row i+1: x  y0  y1

Above is an attempt to show that for x in the range x[i] to x[i+1], the
interpolation is between y1[i] and y0[i+1].  So, y0[0] and y1[-1] are
never used.

"""



cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 1.0),
                   (1.0, 0.1, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (0.25,0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75,1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25,0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75,0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25,1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75,0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

# Make a modified version of cdict3 with some transparency
# in the middle of the range.
cdict4 = cdict3.copy()
cdict4['alpha'] = ((0.0, 1.0, 1.0),
                #   (0.25,1.0, 1.0),
                   (0.5, 0.3, 0.3),
                #   (0.75,1.0, 1.0),
                   (1.0, 1.0, 1.0))


# Now we will use this example to illustrate 3 ways of
# handling custom colormaps.
# First, the most direct and explicit:

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

# Second, create the map explicitly and register it.
# Like the first method, this method works with any kind
# of Colormap, not just
# a LinearSegmentedColormap:

blue_red2 = LinearSegmentedColormap('BlueRed2', cdict2)
plt.register_cmap(cmap=blue_red2)

# Third, for LinearSegmentedColormap only,
# leave everything to register_cmap:

plt.register_cmap(name='BlueRed3', data=cdict3) # optional lut kwarg
plt.register_cmap(name='BlueRedAlpha', data=cdict4)

# Make some illustrative fake data:

x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2*np.pi, 0.1)
X, Y = np.meshgrid(x,y)
Z = np.cos(X) * np.sin(Y) * 10

# Make the figure:

plt.figure(figsize=(6,9))
plt.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)

# Make 4 subplots:

plt.subplot(2,2,1)
plt.imshow(Z, interpolation='nearest', cmap=blue_red1)
plt.colorbar()

plt.subplot(2,2,2)
cmap = plt.get_cmap('BlueRed2')
plt.imshow(Z, interpolation='nearest', cmap=cmap)
plt.colorbar()

# Now we will set the third cmap as the default.  One would
# not normally do this in the middle of a script like this;
# it is done here just to illustrate the method.

plt.rcParams['image.cmap'] = 'BlueRed3'

plt.subplot(2,2,3)
plt.imshow(Z, interpolation='nearest')
plt.colorbar()
plt.title("Alpha = 1")

# Or as yet another variation, we can replace the rcParams
# specification *before* the imshow with the following *after*
# imshow.
# This sets the new default *and* sets the colormap of the last
# image-like item plotted via pyplot, if any.
#

plt.subplot(2,2,4)
# Draw a line with low zorder so it will be behind the image.
plt.plot([0, 10*np.pi], [0, 20*np.pi], color='c', lw=20, zorder=-1)

plt.imshow(Z, interpolation='nearest')
plt.colorbar()

# Here it is: changing the colormap for the current image and its
# colorbar after they have been plotted.
plt.set_cmap('BlueRedAlpha')
plt.title("Varying alpha")
#

plt.suptitle('Custom Blue-Red colormaps', fontsize=16)

plt.show()





********************************************
./pylab_examples/matplotlib_icon.py
#!/usr/bin/env python
"""
make the matplotlib svg minimization icon
"""
import matplotlib
#matplotlib.use('Svg')
from pylab import *

rc('grid', ls='-', lw=2, color='k')
fig = figure(figsize=(1, 1), dpi=72)
axes([0.025, 0.025, 0.95, 0.95], axisbg='#bfd1d4')

t = arange(0, 2, 0.05)
s = sin(2*pi*t)
plot(t,s, linewidth=4, color="#ca7900")
axis([-.2, 2.2, -1.2, 1.2])

# grid(True)
setp(gca(), xticklabels=[], yticklabels=[])
savefig('matplotlib', facecolor='0.75')





********************************************
./pylab_examples/color_demo.py
#!/usr/bin/env python
"""
matplotlib gives you 4 ways to specify colors,

    1) as a single letter string, ala MATLAB

    2) as an html style hex string or html color name

    3) as an R,G,B tuple, where R,G,B, range from 0-1

    4) as a string representing a floating point number
       from 0 to 1, corresponding to shades of gray.

See help(colors) for more info.
"""
from pylab import *

subplot(111, axisbg='darkslategray')
#subplot(111, axisbg='#ababab')
t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s, 'y')
xlabel('time (s)', color='r')
ylabel('voltage (mV)', color='0.5') # grayscale color
title('About as silly as it gets, folks', color='#afeeee')
show()




********************************************
./pylab_examples/geo_demo.py
import numpy as np
#np.seterr("raise")

from pylab import *

subplot(221, projection="aitoff")
title("Aitoff")
grid(True)

subplot(222, projection="hammer")
title("Hammer")
grid(True)

subplot(223, projection="lambert")
title("Lambert")
grid(True)

subplot(224, projection="mollweide")
title("Mollweide")
grid(True)

show()




********************************************
./pylab_examples/contourf_log.py
'''
Demonstrate use of a log color scale in contourf
'''

from matplotlib import pyplot as P
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal

N = 100
x = np.linspace(-3.0, 3.0, N)
y = np.linspace(-2.0, 2.0, N)

X, Y = np.meshgrid(x, y)

# A low hump with a spike coming out of the top right.
# Needs to have z/colour axis on a log scale so we see both hump and spike.
# linear scale only shows the spike.
z = (bivariate_normal(X, Y, 0.1, 0.2, 1.0, 1.0)
        + 0.1 * bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0))

# Put in some negative values (lower left corner) to cause trouble with logs:
z[:5, :5] = -1

# The following is not strictly essential, but it will eliminate
# a warning.  Comment it out to see the warning.
z = ma.masked_where(z<= 0, z)


# Automatic selection of levels works; setting the
# log locator tells contourf to use a log scale:
cs = P.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)

# Alternatively, you can manually set the levels
# and the norm:
#lev_exp = np.arange(np.floor(np.log10(z.min())-1),
#                       np.ceil(np.log10(z.max())+1))
#levs = np.power(10, lev_exp)
#cs = P.contourf(X, Y, z, levs, norm=colors.LogNorm())

#The 'extend' kwarg does not work yet with a log scale.

cbar = P.colorbar()

P.show()






********************************************
./pylab_examples/barb_demo.py
'''
Demonstration of wind barb plots
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 5)
X,Y = np.meshgrid(x, x)
U, V = 12*X, 12*Y

data = [(-1.5, .5, -6, -6),
        (1, -1, -46, 46),
        (-3, -1, 11, -11),
        (1, 1.5, 80, 80),
        (0.5, 0.25, 25, 15),
        (-1.5, -0.5, -5, 40)]

data = np.array(data, dtype=[('x', np.float32), ('y', np.float32),
    ('u', np.float32), ('v', np.float32)])

#Default parameters, uniform grid
ax = plt.subplot(2,2,1)
ax.barbs(X, Y, U, V)

#Arbitrary set of vectors, make them longer and change the pivot point
#(point around which they're rotated) to be the middle
ax = plt.subplot(2,2,2)
ax.barbs(data['x'], data['y'], data['u'], data['v'], length=8, pivot='middle')

#Showing colormapping with uniform grid.  Fill the circle for an empty barb,
#don't round the values, and change some of the size parameters
ax = plt.subplot(2,2,3)
ax.barbs(X, Y, U, V, np.sqrt(U*U + V*V), fill_empty=True, rounding=False,
    sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3))

#Change colors as well as the increments for parts of the barbs
ax = plt.subplot(2,2,4)
ax.barbs(data['x'], data['y'], data['u'], data['v'], flagcolor='r',
    barbcolor=['b','g'], barb_increments=dict(half=10, full=20, flag=100),
    flip_barb=True)

#Masked arrays are also supported
masked_u = np.ma.masked_array(data['u'])
masked_u[4] = 1000 #Bad value that should not be plotted when masked
masked_u[4] = np.ma.masked

#Identical plot to panel 2 in the first figure, but with the point at
#(0.5, 0.25) missing (masked)
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.barbs(data['x'], data['y'], masked_u, data['v'], length=8, pivot='middle')

plt.show()




********************************************
./pylab_examples/annotation_demo2.py

from matplotlib.pyplot import figure, show
from matplotlib.patches import Ellipse
import numpy as np

if 1:
    fig = figure(1,figsize=(8,5))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-4,3))

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2*np.pi*t)
    line, = ax.plot(t, s, lw=3, color='purple')

    ax.annotate('arrowstyle', xy=(0, 1),  xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )

    ax.annotate('arc3', xy=(0.5, -1),  xycoords='data',
                xytext=(-30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )

    ax.annotate('arc', xy=(1., 1),  xycoords='data',
                xytext=(-40, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=30,rad=10"),
                )

    ax.annotate('arc', xy=(1.5, -1),  xycoords='data',
                xytext=(-40, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"),
                )

    ax.annotate('angle', xy=(2., 1),  xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )

    ax.annotate('angle3', xy=(2.5, -1),  xycoords='data',
                xytext=(-50, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-90"),
                )


    ax.annotate('angle', xy=(3., 1),  xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )

    ax.annotate('angle', xy=(3.5, -1),  xycoords='data',
                xytext=(-70, -60), textcoords='offset points',
                size=20,
                bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=-90,rad=10"),
                )

    ax.annotate('angle', xy=(4., 1),  xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                                shrinkA=0, shrinkB=10,
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )


    ann = ax.annotate('', xy=(4., 1.),  xycoords='data',
                xytext=(4.5, -1), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="bar",
                                ec="k",
                                shrinkA=5, shrinkB=5,
                                )
                )


if 1:
    fig = figure(2)
    fig.clf()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-5,3))

    el = Ellipse((2, -1), 0.5, 0.5)
    ax.add_patch(el)

    ax.annotate('$->$', xy=(2., -1),  xycoords='data',
                xytext=(-150, -140), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                                patchB=el,
                                connectionstyle="angle,angleA=90,angleB=0,rad=10"),
                )

    ax.annotate('fancy', xy=(2., -1),  xycoords='data',
                xytext=(-100, 60), textcoords='offset points',
                size=20,
                #bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="fancy",
                                fc="0.6", ec="none",
                                patchB=el,
                                connectionstyle="angle3,angleA=0,angleB=-90"),
                )

    ax.annotate('simple', xy=(2., -1),  xycoords='data',
                xytext=(100, 60), textcoords='offset points',
                size=20,
                #bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=el,
                                connectionstyle="arc3,rad=0.3"),
                )

    ax.annotate('wedge', xy=(2., -1),  xycoords='data',
                xytext=(-100, -100), textcoords='offset points',
                size=20,
                #bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                                fc="0.6", ec="none",
                                patchB=el,
                                connectionstyle="arc3,rad=-0.3"),
                )


    ann = ax.annotate('wedge', xy=(2., -1),  xycoords='data',
                xytext=(0, -45), textcoords='offset points',
                size=20,
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)),
                arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                patchA=None,
                                patchB=el,
                                relpos=(0.2, 0.8),
                                connectionstyle="arc3,rad=-0.1"),
                )

    ann = ax.annotate('wedge', xy=(2., -1),  xycoords='data',
                xytext=(35, 0), textcoords='offset points',
                size=20, va="center",
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
                arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                fc=(1.0, 0.7, 0.7), ec="none",
                                patchA=None,
                                patchB=el,
                                relpos=(0.2, 0.5),
                                )
                )


show()




********************************************
./pylab_examples/vline_hline_demo.py
#!/usr/bin/env python

"""
Small demonstration of the hlines and vlines plots.
"""

from matplotlib import pyplot as plt
from numpy import sin, exp,  absolute, pi, arange
from numpy.random import normal


def f(t):
    s1 = sin(2 * pi * t)
    e1 = exp(-t)
    return absolute((s1 * e1)) + .05


t = arange(0.0, 5.0, 0.1)
s = f(t)
nse = normal(0.0, 0.3, t.shape) * s

fig = plt.figure(figsize=(12, 6))
vax = fig.add_subplot(121)
hax = fig.add_subplot(122)

vax.plot(t, s + nse, 'b^')
vax.vlines(t, [0], s)
vax.set_xlabel('time (s)')
vax.set_title('Vertical lines demo')

hax.plot(s + nse, t, 'b^')
hax.hlines(t, [0], s, lw=2)
hax.set_xlabel('time (s)')
hax.set_title('Horizontal lines demo')

plt.show()




********************************************
./pylab_examples/patheffect_demo.py
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

if 1:
    plt.figure(1, figsize=(8,3))
    ax1 = plt.subplot(131)
    ax1.imshow([[1,2],[2,3]])
    txt = ax1.annotate("test", (1., 1.), (0., 0),
                       arrowprops=dict(arrowstyle="->",
                                       connectionstyle="angle3", lw=2),
                       size=20, ha="center", path_effects=[PathEffects.withStroke(linewidth=3,
                                                 foreground="w")])
    txt.arrow_patch.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])

    ax1.grid(True, linestyle="-")

    pe = [PathEffects.withStroke(linewidth=3,
                                 foreground="w")]
    for l in ax1.get_xgridlines() + ax1.get_ygridlines():
        l.set_path_effects(pe)

    ax2 = plt.subplot(132)
    arr = np.arange(25).reshape((5,5))
    ax2.imshow(arr)
    cntr = ax2.contour(arr, colors="k")

    plt.setp(cntr.collections, path_effects=[
        PathEffects.withStroke(linewidth=3, foreground="w")])

    clbls = ax2.clabel(cntr, fmt="%2.0f", use_clabeltext=True)
    plt.setp(clbls, path_effects=[
        PathEffects.withStroke(linewidth=3, foreground="w")])

    # shadow as a path effect
    ax3 = plt.subplot(133)
    p1, = ax3.plot([0, 1], [0, 1])
    leg = ax3.legend([p1], ["Line 1"], fancybox=True, loc=2)
    leg.legendPatch.set_path_effects([PathEffects.withSimplePatchShadow()])

    plt.show()




********************************************
./pylab_examples/log_test.py
#!/usr/bin/env python
from pylab import *

dt = 0.01
t = arange(dt, 20.0, dt)

semilogx(t, exp(-t/5.0))
grid(True)

show()




********************************************
./pylab_examples/figlegend_demo.py
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.7])
ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.7])

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2*np.pi*x)
y2 = np.exp(-x)
l1, l2 = ax1.plot(x, y1, 'rs-', x, y2, 'go')

y3 = np.sin(4*np.pi*x)
y4 = np.exp(-2*x)
l3, l4 = ax2.plot(x, y3, 'yd-', x, y3, 'k^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
plt.show()




********************************************
./pylab_examples/usetex_demo.py
import matplotlib
matplotlib.rc('text', usetex = True)
import pylab
import numpy as np

## interface tracking profiles
N = 500
delta = 0.6
X = -1 + 2. * np.arange(N) / (N - 1)
pylab.plot(X, (1 - np.tanh(4. * X / delta)) / 2,     ## phase field tanh profiles
           X, (X + 1) / 2,                           ## level set distance function
           X, (1.4 + np.tanh(4. * X / delta)) / 4,   ## composition profile
           X, X < 0, 'k--',                               ## sharp interface
           linewidth = 5)

## legend
pylab.legend((r'phase field', r'level set', r'composition', r'sharp interface'), shadow = True, loc = (0.01, 0.55))

ltext = pylab.gca().get_legend().get_texts()
pylab.setp(ltext[0], fontsize = 20, color = 'b')
pylab.setp(ltext[1], fontsize = 20, color = 'g')
pylab.setp(ltext[2], fontsize = 20, color = 'r')
pylab.setp(ltext[3], fontsize = 20, color = 'k')

## the arrow
height = 0.1
offset = 0.02
pylab.plot((-delta / 2., delta / 2), (height, height), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.text(-0.06, height - 0.06, r'$\delta$', {'color' : 'k', 'fontsize' : 24})

## X-axis label
pylab.xticks((-1, 0,  1), ('-1', '0',  '1'), color = 'k', size = 20)

## Left Y-axis labels
pylab.ylabel(r'\bf{phase field} $\phi$', {'color'    : 'b',
                                   'fontsize'   : 20 })
pylab.yticks((0, 0.5, 1), ('0', '.5', '1'), color = 'k', size = 20)

## Right Y-axis labels
pylab.text(1.05, 0.5, r"\bf{level set} $\phi$", {'color' : 'g', 'fontsize' : 20},
           horizontalalignment = 'left',
           verticalalignment = 'center',
           rotation = 90,
           clip_on = False)
pylab.text(1.01, -0.02, "-1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.98, "1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.48, "0", {'color' : 'k', 'fontsize' : 20})

## level set equations
pylab.text(0.1, 0.85, r'$|\nabla\phi| = 1,$ \newline $ \frac{\partial \phi}{\partial t} + U|\nabla \phi| = 0$', {'color' : 'g', 'fontsize' : 20})

## phase field equations
pylab.text(0.2, 0.15, r'$\mathcal{F} = \int f\left( \phi, c \right) dV,$ \newline $ \frac{ \partial \phi } { \partial t } = -M_{ \phi } \frac{ \delta \mathcal{F} } { \delta \phi }$', {'color' : 'b', 'fontsize' : 20})

## these went wrong in pdf in a previous version
pylab.text(-.9,.42,r'gamma: $\gamma$', {'color': 'r', 'fontsize': 20})
pylab.text(-.9,.36,r'Omega: $\Omega$', {'color': 'b', 'fontsize': 20})

pylab.show()




********************************************
./pylab_examples/spectrum_demo.py
#!/usr/bin/env python
# python

from pylab import *

dt = 0.01
Fs = 1/dt
t = arange(0, 10, dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]
s = 0.1*sin(2*pi*t) + cnse

subplot(3, 2, 1)
plot(t, s)

subplot(3, 2, 3)
magnitude_spectrum(s, Fs=Fs)

subplot(3, 2, 4)
magnitude_spectrum(s, Fs=Fs, scale='dB')

subplot(3, 2, 5)
angle_spectrum(s, Fs=Fs)

subplot(3, 2, 6)
phase_spectrum(s, Fs=Fs)

show()




********************************************
./pylab_examples/dolphin.py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import numpy as np


r = np.random.rand(50)
t = np.random.rand(50) * np.pi * 2.0
x = r * np.cos(t)
y = r * np.sin(t)

fig, ax = plt.subplots(figsize=(6,6))
circle = Circle((0, 0), 1, facecolor='none',
                edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)

im = plt.imshow(np.random.random((100, 100)),
                origin='lower', cmap=cm.winter,
                interpolation='spline36',
                extent=([-1, 1, -1, 1]))
im.set_clip_path(circle)

plt.plot(x, y, 'o', color=(0.9, 0.9, 1.0), alpha=0.8)

# Dolphin from OpenClipart library by Andy Fitzsimon
#       <cc:License rdf:about="http://web.resource.org/cc/PublicDomain">
#         <cc:permits rdf:resource="http://web.resource.org/cc/Reproduction"/>
#         <cc:permits rdf:resource="http://web.resource.org/cc/Distribution"/>
#         <cc:permits rdf:resource="http://web.resource.org/cc/DerivativeWorks"/>
#       </cc:License>

dolphin = """
M -0.59739425,160.18173 C -0.62740401,160.18885 -0.57867129,160.11183
-0.57867129,160.11183 C -0.57867129,160.11183 -0.5438361,159.89315
-0.39514638,159.81496 C -0.24645668,159.73678 -0.18316813,159.71981
-0.18316813,159.71981 C -0.18316813,159.71981 -0.10322971,159.58124
-0.057804323,159.58725 C -0.029723983,159.58913 -0.061841603,159.60356
-0.071265813,159.62815 C -0.080250183,159.65325 -0.082918513,159.70554
-0.061841203,159.71248 C -0.040763903,159.7194 -0.0066711426,159.71091
0.077336307,159.73612 C 0.16879567,159.76377 0.28380306,159.86448
0.31516668,159.91533 C 0.3465303,159.96618 0.5011127,160.1771
0.5011127,160.1771 C 0.63668998,160.19238 0.67763022,160.31259
0.66556395,160.32668 C 0.65339985,160.34212 0.66350443,160.33642
0.64907098,160.33088 C 0.63463742,160.32533 0.61309688,160.297
0.5789627,160.29339 C 0.54348657,160.28968 0.52329693,160.27674
0.50728856,160.27737 C 0.49060916,160.27795 0.48965803,160.31565
0.46114204,160.33673 C 0.43329696,160.35786 0.4570711,160.39871
0.43309565,160.40685 C 0.4105108,160.41442 0.39416631,160.33027
0.3954995,160.2935 C 0.39683269,160.25672 0.43807996,160.21522
0.44567915,160.19734 C 0.45327833,160.17946 0.27946869,159.9424
-0.061852613,159.99845 C -0.083965233,160.0427 -0.26176109,160.06683
-0.26176109,160.06683 C -0.30127962,160.07028 -0.21167141,160.09731
-0.24649368,160.1011 C -0.32642366,160.11569 -0.34521187,160.06895
-0.40622293,160.0819 C -0.467234,160.09485 -0.56738444,160.17461
-0.59739425,160.18173
"""

vertices = []
codes = []
parts = dolphin.split()
i = 0
code_map = {
    'M': (Path.MOVETO, 1),
    'C': (Path.CURVE4, 3),
    'L': (Path.LINETO, 1)
    }

while i < len(parts):
    code = parts[i]
    path_code, npoints = code_map[code]
    codes.extend([path_code] * npoints)
    vertices.extend([[float(x) for x in y.split(',')] for y in parts[i+1:i+npoints+1]])
    i += npoints + 1
vertices = np.array(vertices, np.float)
vertices[:,1] -= 160

dolphin_path = Path(vertices, codes)
dolphin_patch = PathPatch(dolphin_path, facecolor=(0.6, 0.6, 0.6),
                          edgecolor=(0.0, 0.0, 0.0))
ax.add_patch(dolphin_patch)

vertices = Affine2D().rotate_deg(60).transform(vertices)
dolphin_path2 = Path(vertices, codes)
dolphin_patch2 = PathPatch(dolphin_path2, facecolor=(0.5, 0.5, 0.5),
                           edgecolor=(0.0, 0.0, 0.0))
ax.add_patch(dolphin_patch2)

plt.show()




********************************************
./pylab_examples/subplot_toolbar.py
#!/usr/bin/env python

from pylab import *

fig = figure()
subplot(221)
imshow(rand(100,100))
subplot(222)
imshow(rand(100,100))
subplot(223)
imshow(rand(100,100))
subplot(224)
imshow(rand(100,100))

subplot_tool()
show()




********************************************
./pylab_examples/multiline.py
#!/usr/bin/env python
from pylab import *
#from matplotlib.pyplot import *
#from numpy import arange

if 1:
    figure(figsize=(7, 4))
    ax = subplot(121)
    ax.set_aspect(1)
    plot(arange(10))
    xlabel('this is a xlabel\n(with newlines!)')
    ylabel('this is vertical\ntest', multialignment='center')
    #ylabel('this is another!')
    text(2, 7,'this is\nyet another test',
         rotation=45,
         horizontalalignment = 'center',
         verticalalignment   = 'top',
         multialignment      = 'center')

    grid(True)



    subplot(122)
    
    text(0.29, 0.7, "Mat\nTTp\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

    text(0.34, 0.7, "Mag\nTTT\n123", size=18,
         va="baseline", ha="left", multialignment="left",
         bbox=dict(fc="none"))

    text(0.95, 0.7, "Mag\nTTT$^{A^A}$\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

    xticks([0.2, 0.4, 0.6, 0.8, 1.],
           ["Jan\n2009","Feb\n2009","Mar\n2009", "Apr\n2009", "May\n2009"])
    
    axhline(0.7)
    title("test line spacing for multiline text")

subplots_adjust(bottom=0.25, top=0.8)
draw()
show()




********************************************
./pylab_examples/tex_unicode_demo.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This demo is tex_demo.py modified to have unicode. See that file for
more information.
"""
from __future__ import unicode_literals
import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
from numpy import arange, cos, pi
from matplotlib.pyplot import (figure, axes, plot, xlabel, ylabel, title,
     grid, savefig, show)

figure(1, figsize=(6,4))
ax = axes([0.1, 0.1, 0.8, 0.7])
t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)+2
plot(t, s)

xlabel(r'\textbf{time (s)}')
ylabel(r'\textit{Velocity (\u00B0/sec)}', fontsize=16)
title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
      fontsize=16, color='r')
grid(True)
show()




********************************************
./pylab_examples/hatch_demo.py
"""
Hatching (pattern filled polygons) is supported currently in the PS,
PDF, SVG and Agg backends only.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.bar(range(1,5), range(1,5), color='red', edgecolor='black', hatch="/")
ax1.bar(range(1,5), [6] * 4, bottom=range(1,5), color='blue', edgecolor='black', hatch='//')
ax1.set_xticks([1.5,2.5,3.5,4.5])

ax2 = fig.add_subplot(132)
bars = ax2.bar(range(1,5), range(1,5), color='yellow', ecolor='black') + \
    ax2.bar(range(1, 5), [6] * 4, bottom=range(1,5), color='green', ecolor='black')
ax2.set_xticks([1.5,2.5,3.5,4.5])

patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)

ax3 = fig.add_subplot(133)
ax3.fill([1,3,3,1],[1,1,2,2], fill=False, hatch='\\')
ax3.add_patch(Ellipse((4,1.5), 4, 0.5, fill=False, hatch='*'))
ax3.add_patch(Polygon([[0,0],[4,1.1],[6,2.5],[2,1.4]], closed=True,
                      fill=False, hatch='/'))
ax3.set_xlim((0,6))
ax3.set_ylim((0,2.5))

plt.show()




********************************************
./pylab_examples/fancytextbox_demo.py
import matplotlib.pyplot as plt

plt.text(0.6, 0.5, "test", size=50, rotation=30.,
         ha="center", va="center", 
         bbox = dict(boxstyle="round",
                     ec=(1., 0.5, 0.5),
                     fc=(1., 0.8, 0.8),
                     )
         )

plt.text(0.5, 0.4, "test", size=50, rotation=-30.,
         ha="right", va="top", 
         bbox = dict(boxstyle="square",
                     ec=(1., 0.5, 0.5),
                     fc=(1., 0.8, 0.8),
                     )
         )

    
plt.draw()
plt.show()




********************************************
./pylab_examples/accented_text.py
#!/usr/bin/env python
"""
matplotlib supports accented characters via TeX mathtext

The following accents are provided: \hat, \breve, \grave, \bar,
\acute, \tilde, \vec, \dot, \ddot.  All of them have the same syntax,
eg to make an overbar you do \bar{o} or to make an o umlaut you do
\ddot{o}.  The shortcuts are also provided, eg: \"o \'e \`e \~n \.x
\^y

"""
from pylab import *

axes([0.1, 0.15, 0.8, 0.75])
plot(range(10))

title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{i}\bar{A}\tilde{n}\vec{q}$', fontsize=20)
# shorthand is also supported and curly's are optional
xlabel(r"""$\"o\ddot o \'e\`e\~n\.x\^y$""", fontsize=20)


show()




********************************************
./pylab_examples/eventplot_demo.py
#!/usr/bin/env python
# -*- Coding:utf-8 -*-
'''an eventplot showing sequences of events with various line properties
the plot is shown in both horizontal and vertical orientations'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 8.0

# set the random seed
np.random.seed(0)

# create random data
data1 = np.random.random([6, 50])

# set different colors for each set of positions
colors1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]])

# set different line properties for each set of positions
# note that some overlap
lineoffsets1 = np.array([-15, -3, 1, 1.5, 6, 10])
linelengths1 = [5, 2, 1, 1, 3, 1.5]

fig = plt.figure()

# create a horizontal plot
ax1 = fig.add_subplot(221)
ax1.eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
              linelengths=linelengths1)


# create a vertical plot
ax2 = fig.add_subplot(223)
ax2.eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
              linelengths=linelengths1, orientation='vertical')

# create another set of random data.
# the gamma distribution is only used fo aesthetic purposes
data2 = np.random.gamma(4, size=[60, 50])

# use individual values for the parameters this time
# these values will be used for all data sets (except lineoffsets2, which
# sets the increment between each data set in this usage)
colors2 = [[0, 0, 0]]
lineoffsets2 = 1
linelengths2 = 1

# create a horizontal plot
ax1 = fig.add_subplot(222)
ax1.eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
              linelengths=linelengths2)


# create a vertical plot
ax2 = fig.add_subplot(224)
ax2.eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
              linelengths=linelengths2, orientation='vertical')

plt.show()




********************************************
./pylab_examples/line_styles.py
#!/usr/bin/env python
# This should probably be replaced with a demo that shows all
# line and marker types in a single panel, with labels.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

t = np.arange(0.0, 1.0, 0.1)
s = np.sin(2*np.pi*t)
linestyles = ['_', '-', '--', ':']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

plt.figure(figsize=(8,8))

axisNum = 0
for row in range(6):
    for col in range(5):
        axisNum += 1
        ax = plt.subplot(6, 5, axisNum)
        color = colors[axisNum % len(colors)]
        if axisNum < len(linestyles):
            plt.plot(t, s, linestyles[axisNum], color=color, markersize=10)
        else:
            style = styles[(axisNum - len(linestyles)) % len(styles)]
            plt.plot(t, s, linestyle='None', marker=style, color=color, markersize=10)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

plt.show()




********************************************
./pylab_examples/interp_demo.py
import matplotlib.pyplot as plt
from numpy import pi, sin, linspace
from matplotlib.mlab import stineman_interp

x = linspace(0,2*pi,20);
y = sin(x); yp = None
xi = linspace(x[0],x[-1],100);
yi = stineman_interp(xi,x,y,yp);

fig, ax = plt.subplots()
ax.plot(x,y,'ro',xi,yi,'-b.')
plt.show()





********************************************
./pylab_examples/boxplot_demo.py
#!/usr/bin/python

#
# Example boxplot code
#

from pylab import *

# fake up some data
spread= rand(50) * 100
center = ones(25) * 50
flier_high = rand(10) * 100 + 100
flier_low = rand(10) * -100
data =concatenate((spread, center, flier_high, flier_low), 0)

# basic plot
boxplot(data)

# notched plot
figure()
boxplot(data,1)

# change outlier point symbols
figure()
boxplot(data,0,'gD')

# don't show outlier points
figure()
boxplot(data,0,'')

# horizontal boxes
figure()
boxplot(data,0,'rs',0)

# change whisker length
figure()
boxplot(data,0,'rs',0,0.75)

# fake up some more data
spread= rand(50) * 100
center = ones(25) * 40
flier_high = rand(10) * 100 + 100
flier_low = rand(10) * -100
d2 = concatenate( (spread, center, flier_high, flier_low), 0 )
data.shape = (-1, 1)
d2.shape = (-1, 1)
#data = concatenate( (data, d2), 1 )
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [data, d2, d2[::2,0]]
# multiple box plots on one figure
figure()
boxplot(data)

show()





********************************************
./pylab_examples/text_rotation.py
#!/usr/bin/env python
"""
The way matplotlib does text layout is counter-intuitive to some, so
this example is designed to make it a little clearer.  The text is
aligned by it's bounding box (the rectangular box that surrounds the
ink rectangle).  The order of operations is basically rotation then
alignment, rather than alignment then rotation.  Basically, the text
is centered at your x,y location, rotated around this point, and then
aligned according to the bounding box of the rotated text.

So if you specify left, bottom alignment, the bottom left of the
bounding box of the rotated text will be at the x,y coordinate of the
text.

But a picture is worth a thousand words!
"""
from pylab import *

def addtext(props):
    text(0.5, 0.5, 'text 0', props, rotation=0)
    text(1.5, 0.5, 'text 45', props, rotation=45)
    text(2.5, 0.5, 'text 135', props, rotation=135)
    text(3.5, 0.5, 'text 225', props, rotation=225)
    text(4.5, 0.5, 'text -45', props, rotation=-45)
    yticks([0,.5,1])
    grid(True)

# the text bounding box
bbox = {'fc':'0.8', 'pad':0}

subplot(211)
addtext({'ha':'center', 'va':'center', 'bbox':bbox})
xlim(0,5)
xticks(arange(0, 5.1, 0.5), [])
ylabel('center / center')
subplot(212)
addtext({'ha':'left', 'va':'bottom', 'bbox':bbox})
xlim(0,5)
xticks(arange(0, 5.1, 0.5))
ylabel('left / bottom')
show()




********************************************
./pylab_examples/triplot_demo.py
"""
Creating and plotting unstructured triangular grids.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math

# Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.

# First create the x and y coordinates of the points.
n_angles = 36
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

# Plot the triangulation.
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(triang, 'bo-')
plt.title('triplot of Delaunay triangulation')


# You can specify your own triangulation rather than perform a Delaunay
# triangulation of the points, where each triangle is given by the indices of
# the three points that make up the triangle, ordered in either a clockwise or
# anticlockwise manner.

xy = np.asarray([
    [-0.101, 0.872], [-0.080, 0.883], [-0.069, 0.888], [-0.054, 0.890],
    [-0.045, 0.897], [-0.057, 0.895], [-0.073, 0.900], [-0.087, 0.898],
    [-0.090, 0.904], [-0.069, 0.907], [-0.069, 0.921], [-0.080, 0.919],
    [-0.073, 0.928], [-0.052, 0.930], [-0.048, 0.942], [-0.062, 0.949],
    [-0.054, 0.958], [-0.069, 0.954], [-0.087, 0.952], [-0.087, 0.959],
    [-0.080, 0.966], [-0.085, 0.973], [-0.087, 0.965], [-0.097, 0.965],
    [-0.097, 0.975], [-0.092, 0.984], [-0.101, 0.980], [-0.108, 0.980],
    [-0.104, 0.987], [-0.102, 0.993], [-0.115, 1.001], [-0.099, 0.996],
    [-0.101, 1.007], [-0.090, 1.010], [-0.087, 1.021], [-0.069, 1.021],
    [-0.052, 1.022], [-0.052, 1.017], [-0.069, 1.010], [-0.064, 1.005],
    [-0.048, 1.005], [-0.031, 1.005], [-0.031, 0.996], [-0.040, 0.987],
    [-0.045, 0.980], [-0.052, 0.975], [-0.040, 0.973], [-0.026, 0.968],
    [-0.020, 0.954], [-0.006, 0.947], [ 0.003, 0.935], [ 0.006, 0.926],
    [ 0.005, 0.921], [ 0.022, 0.923], [ 0.033, 0.912], [ 0.029, 0.905],
    [ 0.017, 0.900], [ 0.012, 0.895], [ 0.027, 0.893], [ 0.019, 0.886],
    [ 0.001, 0.883], [-0.012, 0.884], [-0.029, 0.883], [-0.038, 0.879],
    [-0.057, 0.881], [-0.062, 0.876], [-0.078, 0.876], [-0.087, 0.872],
    [-0.030, 0.907], [-0.007, 0.905], [-0.057, 0.916], [-0.025, 0.933],
    [-0.077, 0.990], [-0.059, 0.993]])
x = xy[:, 0]*180/3.14159
y = xy[:, 1]*180/3.14159

triangles = np.asarray([
    [67, 66,  1], [65,  2, 66], [ 1, 66,  2], [64,  2, 65], [63,  3, 64],
    [60, 59, 57], [ 2, 64,  3], [ 3, 63,  4], [ 0, 67,  1], [62,  4, 63],
    [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [ 4, 62, 68],
    [ 6,  5,  9], [61, 68, 62], [69, 68, 61], [ 9,  5, 70], [ 6,  8,  7],
    [ 4, 70,  5], [ 8,  6,  9], [56, 69, 57], [69, 56, 52], [70, 10,  9],
    [54, 53, 55], [56, 55, 53], [68, 70,  4], [52, 56, 53], [11, 10, 12],
    [69, 71, 68], [68, 13, 70], [10, 70, 13], [51, 50, 52], [13, 68, 71],
    [52, 71, 69], [12, 10, 13], [71, 52, 50], [71, 14, 13], [50, 49, 71],
    [49, 48, 71], [14, 16, 15], [14, 71, 48], [17, 19, 18], [17, 20, 19],
    [48, 16, 14], [48, 47, 16], [47, 46, 16], [16, 46, 45], [23, 22, 24],
    [21, 24, 22], [17, 16, 45], [20, 17, 45], [21, 25, 24], [27, 26, 28],
    [20, 72, 21], [25, 21, 72], [45, 72, 20], [25, 28, 26], [44, 73, 45],
    [72, 45, 73], [28, 25, 29], [29, 25, 31], [43, 73, 44], [73, 43, 40],
    [72, 73, 39], [72, 31, 25], [42, 40, 43], [31, 30, 29], [39, 73, 40],
    [42, 41, 40], [72, 33, 31], [32, 31, 33], [39, 38, 72], [33, 72, 38],
    [33, 38, 34], [37, 35, 38], [34, 38, 35], [35, 37, 36]])

# Rather than create a Triangulation object, can simply pass x, y and triangles
# arrays to triplot directly.  It would be better to use a Triangulation object
# if the same triangulation was to be used more than once to save duplicated
# calculations.
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(x, y, triangles, 'go-')
plt.title('triplot of user-specified triangulation')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')

plt.show()




********************************************
./pylab_examples/xcorr_demo.py
import matplotlib.pyplot as plt
import numpy as np

x,y = np.random.randn(2,100)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)

plt.show()





********************************************
./pylab_examples/shading_example.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# example showing how to make shaded relief plots 
# like Mathematica
# (http://reference.wolfram.com/mathematica/ref/ReliefPlot.html)
# or Generic Mapping Tools
# (http://gmt.soest.hawaii.edu/gmt/doc/gmt/html/GMT_Docs/node145.html)

# test data
X,Y=np.mgrid[-5:5:0.05,-5:5:0.05]
Z=np.sqrt(X**2+Y**2)+np.sin(X**2+Y**2)
# create light source object.
ls = LightSource(azdeg=0,altdeg=65)
# shade data, creating an rgb array.
rgb = ls.shade(Z,plt.cm.copper)
# plot un-shaded and shaded images.
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(Z,cmap=plt.cm.copper)
plt.title('imshow')
plt.xticks([]); plt.yticks([])
plt.subplot(122)
plt.imshow(rgb)
plt.title('imshow with shading')
plt.xticks([]); plt.yticks([])
plt.show()




********************************************
./pylab_examples/psd_demo3.py
#This is a ported version of a MATLAB example from the signal processing
#toolbox that showed some difference at one time between Matplotlib's and
#MATLAB's scaling of the PSD.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

fs = 1000
t = np.linspace(0, 0.3, 301)
A = np.array([2, 8]).reshape(-1, 1)
f = np.array([150, 140]).reshape(-1, 1)
xn = (A * np.sin(2 * np.pi * f * t)).sum(axis=0) + 5 * np.random.randn(*t.shape)

yticks = np.arange(-50, 30, 10)
xticks = np.arange(0,550,100)
plt.subplots_adjust(hspace=0.45, wspace=0.3)
plt.subplot(1,2,1)

plt.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
    scale_by_freq=True)
plt.title('Periodogram')
plt.yticks(yticks)
plt.xticks(xticks)
plt.grid(True)

plt.subplot(1,2,2)
plt.psd(xn, NFFT=150, Fs=fs, window=mlab.window_none, noverlap=75, pad_to=512,
    scale_by_freq=True)
plt.title('Welch')
plt.xticks(xticks)
plt.yticks(yticks)
plt.ylabel('')
plt.grid(True)

plt.show()




********************************************
./pylab_examples/animation_demo.py
"""
Pyplot animation example.

The method shown here is only for very simple, low-performance
use.  For more demanding applications, look at the animation
module and the examples that use it.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
y = np.arange(5)
z = x * y[:,np.newaxis]

for i in xrange(5):
    if i==0:
        p = plt.imshow(z)
        fig = plt.gcf()
        plt.clim()   # clamp the color limits
        plt.title("Boring slide show")
    else:
        z = z + 2
        p.set_data(z)

    print("step", i)
    plt.pause(0.5)




********************************************
./pylab_examples/masked_demo.py
#!/usr/bin/env python
'''
Plot lines with points masked out.

This would typically be used with gappy data, to
break the line at the data gaps.
'''

from pylab import *

x = ma.arange(0, 2*pi, 0.02)
y = ma.sin(x)
y1 = sin(2*x)
y2 = sin(3*x)
ym1 = ma.masked_where(y1 > 0.5, y1)
ym2 = ma.masked_where(y2 < -0.5, y2)

lines = plot(x, y, 'r', x, ym1, 'g', x, ym2, 'bo')
setp(lines[0], linewidth = 4)
setp(lines[1], linewidth = 2)
setp(lines[2], markersize = 10)

legend( ('No mask', 'Masked if > 0.5', 'Masked if < -0.5') ,
        loc = 'upper right')
title('Masked line demo')
show()




********************************************
./pylab_examples/usetex_baseline_test.py

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

from matplotlib import rcParams
rcParams['text.usetex']=True
rcParams['text.latex.unicode']=True

class Axes(maxes.Axes):
    """
    A hackish way to simultaneously draw texts w/ usetex=True and
    usetex=False in the same figure. It does not work in the ps backend.
    """
    def __init__(self, *kl, **kw):
        self.usetex = kw.pop("usetex", "False")
        self.preview = kw.pop("preview", "False")

        maxes.Axes.__init__(self, *kl, **kw)

    def draw(self, renderer):
        usetex = plt.rcParams["text.usetex"]
        preview = plt.rcParams["text.latex.preview"]
        plt.rcParams["text.usetex"] = self.usetex
        plt.rcParams["text.latex.preview"] = self.preview

        maxes.Axes.draw(self, renderer)

        plt.rcParams["text.usetex"] = usetex
        plt.rcParams["text.latex.preview"] = preview

Subplot = maxes.subplot_class_factory(Axes)


def test_window_extent(ax, usetex, preview):

    va = "baseline"
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    #t = ax.text(0., 0., r"mlp", va="baseline", size=150)
    text_kw = dict(va=va,
                   size=50,
                   bbox=dict(pad=0., ec="k", fc="none"))


    test_strings = ["lg", r"$\frac{1}{2}\pi$",
                    r"$p^{3^A}$", r"$p_{3_2}$"]

    ax.axvline(0, color="r")

    for i, s in enumerate(test_strings):

        ax.axhline(i, color="r")
        ax.text(0., 3-i, s, **text_kw)

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-.8,3.9)


    ax.set_title("usetex=%s\npreview=%s" % (str(usetex), str(preview)))



F = plt.figure(figsize=(2.*3,6.5))

for i, usetex, preview in [[0, False, False],
                           [1, True, False],
                           [2, True, True]]:
    ax = Subplot(F, 1, 3, i+1, usetex=usetex, preview=preview)
    F.add_subplot(ax)
    F.subplots_adjust(top=0.85)

    test_window_extent(ax, usetex=usetex, preview=preview)


plt.draw()
plt.show()





********************************************
./pylab_examples/pcolor_demo.py
"""
Demonstrates similarities between pcolor, pcolormesh, imshow and pcolorfast
for drawing quadrilateral grids.

"""
import matplotlib.pyplot as plt
import numpy as np

# make these smaller to increase the resolution
dx, dy = 0.15, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-3, 3 + dy, dy),
                slice(-3, 3 + dx, dx)]
z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()



plt.subplot(2, 2, 1)
plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.title('pcolor')
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()



plt.subplot(2, 2, 2)
plt.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.title('pcolormesh')
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()



plt.subplot(2, 2, 3)
plt.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
           extent=[x.min(), x.max(), y.min(), y.max()],
           interpolation='nearest', origin='lower')
plt.title('image (interp. nearest)')
plt.colorbar()



ax = plt.subplot(2, 2, 4)
ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.title('pcolorfast')
plt.colorbar()



plt.show()



********************************************
./pylab_examples/to_numeric.py
#!/usr/bin/env python
"""
Use backend agg to access the figure canvas as an RGB string and then
convert it to an array and pass the string it to PIL for
rendering
"""

import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from PIL import Image
except ImportError:
    raise SystemExit("PIL must be installed to run this example")

pylab.plot([1,2,3])

canvas = pylab.get_current_fig_manager().canvas

agg = canvas.switch_backends(FigureCanvasAgg)
agg.draw()
s = agg.tostring_rgb()

# get the width and the height to resize the matrix
l,b,w,h = agg.figure.bbox.bounds
w, h = int(w), int(h)


X = pylab.fromstring(s, pylab.uint8)
X.shape = h, w, 3

im = Image.fromstring( "RGB", (w,h), s)

# Uncomment this line to display the image using ImageMagick's
# `display` tool.
# im.show()




********************************************
./pylab_examples/ganged_plots.py
#!/usr/bin/env python
"""
To create plots that share a common axes (visually) you can set the
hspace between the subplots close to zero (do not use zero itself).
Normally you'll want to turn off the tick labels on all but one of the
axes.

In this example the plots share a common xaxis but you can follow the
same logic to supply a common y axis.
"""
from pylab import *

t = arange(0.0, 2.0, 0.01)

s1 = sin(2*pi*t)
s2 = exp(-t)
s3 = s1*s2

# axes rect in relative 0,1 coords left, bottom, width, height.  Turn
# off xtick labels on all but the lower plot


f = figure()
subplots_adjust(hspace=0.001)


ax1 = subplot(311)
ax1.plot(t,s1)
yticks(arange(-0.9, 1.0, 0.4))
ylim(-1,1)

ax2 = subplot(312, sharex=ax1)
ax2.plot(t,s2)
yticks(arange(0.1, 1.0,  0.2))
ylim(0,1)

ax3 = subplot(313, sharex=ax1)
ax3.plot(t,s3)
yticks(arange(-0.9, 1.0, 0.4))
ylim(-1,1)

xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()
setp(xticklabels, visible=False)

show()




********************************************
./pylab_examples/axes_demo.py
#!/usr/bin/env python

from pylab import *

# create some data to use for the plot
dt = 0.001
t = arange(0.0, 10.0, dt)
r = exp(-t[:1000]/0.05)               # impulse response
x = randn(len(t))
s = convolve(x,r)[:len(x)]*dt  # colored noise

# the main axes is subplot(111) by default
plot(t, s)
axis([0, 1, 1.1*amin(s), 2*amax(s) ])
xlabel('time (s)')
ylabel('current (nA)')
title('Gaussian colored noise')

# this is an inset axes over the main axes
a = axes([.65, .6, .2, .2], axisbg='y')
n, bins, patches = hist(s, 400, normed=1)
title('Probability')
setp(a, xticks=[], yticks=[])

# this is another inset axes over the main axes
a = axes([0.2, 0.6, .2, .2], axisbg='y')
plot(t[:len(r)], r)
title('Impulse response')
setp(a, xlim=(0,.2), xticks=[], yticks=[])


show()




********************************************
./pylab_examples/image_clip_path.py
#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

path = Path([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])
patch = PathPatch(path, facecolor='none')
plt.gca().add_patch(patch)

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                origin='lower', extent=[-3,3,-3,3],
                clip_path=patch, clip_on=True)
im.set_clip_path(patch)

plt.show()





********************************************
./pylab_examples/barcode_demo.py
from matplotlib.pyplot import figure, show, cm
from numpy import where
from numpy.random import rand

# the bar
x = where(rand(500)>0.7, 1.0, 0.0)

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=cm.binary, interpolation='nearest')

fig = figure()

# a vertical barcode -- this is broken at present
x.shape = len(x), 1
ax = fig.add_axes([0.1, 0.3, 0.1, 0.6], **axprops)
ax.imshow(x, **barprops)

x = x.copy()
# a horizontal barcode
x.shape = 1, len(x)
ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax.imshow(x, **barprops)


show()





********************************************
./pylab_examples/arctest.py
#!/usr/bin/env python
from pylab import *

def f(t):
    'a damped exponential'
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .2)


l = plot(t1, f(t1), 'ro')
setp(l, 'markersize', 30)
setp(l, 'markerfacecolor', 'b')

show()





********************************************
./pylab_examples/annotation_demo3.py
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)

bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")

ax1.annotate('figure fraction : 0, 0', xy=(0, 0),  xycoords='figure fraction',
             xytext=(20, 20), textcoords='offset points',
             ha="left", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args
             )

ax1.annotate('figure fraction : 1, 1', xy=(1, 1),  xycoords='figure fraction',
             xytext=(-20, -20), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args
             )

ax1.annotate('axes fraction : 0, 0', xy=(0, 0),  xycoords='axes fraction',
             xytext=(20, 20), textcoords='offset points',
             ha="left", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args
             )

ax1.annotate('axes fraction : 1, 1', xy=(1, 1),  xycoords='axes fraction',
             xytext=(-20, -20), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args
             )


an1 = ax1.annotate('Drag me 1', xy=(.5, .7),  xycoords='data',
                   #xytext=(.5, .7), textcoords='data',
                   ha="center", va="center",
                   bbox=bbox_args,
                   #arrowprops=arrow_args
                   )

an2 = ax1.annotate('Drag me 2', xy=(.5, .5),  xycoords=an1,
                   xytext=(.5, .3), textcoords='axes fraction',
                   ha="center", va="center",
                   bbox=bbox_args,
                   arrowprops=dict(patchB=an1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args)
                   )

an3 = ax1.annotate('', xy=(.5, .5),  xycoords=an2,
                   xytext=(.5, .5),  textcoords=an1,
                   ha="center", va="center",
                   bbox=bbox_args,
                   arrowprops=dict(patchA=an1.get_bbox_patch(),
                                   patchB=an2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args)
                   )



t = ax2.annotate('xy=(0, 1)\nxycoords=("data", "axes fraction")',
                 xy=(0, 1),  xycoords=("data", 'axes fraction'),
                 xytext=(0, -20), textcoords='offset points',
                 ha="center", va="top",
                 bbox=bbox_args,
                 arrowprops=arrow_args
                 )

from matplotlib.text import OffsetFrom

ax2.annotate('xy=(0.5, 0)\nxycoords=artist',
             xy=(0.5, 0.),  xycoords=t,
             xytext=(0, -20), textcoords='offset points',
             ha="center", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args
             )

ax2.annotate('xy=(0.8, 0.5)\nxycoords=ax1.transData',
             xy=(0.8, 0.5),  xycoords=ax1.transData,
             xytext=(10, 10), textcoords=OffsetFrom(ax2.bbox, (0, 0), "points"),
             ha="left", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args
             )

ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)

an1.draggable()
an2.draggable()

plt.show()




********************************************
./pylab_examples/hist2d_log_demo.py
from matplotlib.colors import LogNorm
from pylab import *

#normal distribution center at x=0 and y=5
x = randn(100000)
y = randn(100000)+5

hist2d(x, y, bins=40, norm=LogNorm())
colorbar()
show()




********************************************
./pylab_examples/simple_plot.py
from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)
savefig("test.png")
show()




********************************************
./pylab_examples/titles_demo.py
#!/usr/bin/env python
"""
matplotlib can display plot titles centered, flush with the left side of
a set of axes, and flush with the right side of a set of axes.

"""
import matplotlib.pyplot as plt

plt.plot(range(10))

plt.title('Center Title')
plt.title('Left Title', loc='left')
plt.title('Right Title', loc='right')

plt.show()




********************************************
./pylab_examples/line_collection.py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter

import numpy as np

# In order to efficiently plot many lines in a single set of axes,
# Matplotlib has the ability to add the lines all at once. Here is a
# simple example showing how it is done.

x = np.arange(100)
# Here are many sets of y to plot vs x
ys = x[:50, np.newaxis] + x[np.newaxis, :]

segs = np.zeros((50, 100, 2), float)
segs[:,:,1] = ys
segs[:,:,0] = x

# Mask some values to test masked array support:
segs = np.ma.masked_where((segs > 50) & (segs < 60), segs)

# We need to set the plot limits.
ax = plt.axes()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(ys.min(), ys.max())

# colors is sequence of rgba tuples
# linestyle is a string or dash tuple. Legal string values are
#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
#          where onoffseq is an even length tuple of on and off ink in points.
#          If linestyle is omitted, 'solid' is used
# See matplotlib.collections.LineCollection for more information
line_segments = LineCollection(segs,
                                linewidths    = (0.5,1,1.5,2),
                                colors        = [colorConverter.to_rgba(i) \
                                                 for i in ('b','g','r','c','m','y','k')],
                                linestyle = 'solid')
ax.add_collection(line_segments)
ax.set_title('Line collection with masked arrays')
plt.show()






********************************************
./pylab_examples/fancyarrow_demo.py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

styles = mpatches.ArrowStyle.get_styles()

ncol=2
nrow = (len(styles)+1) // ncol
figheight = (nrow+0.5)
fig1 = plt.figure(1, (4.*ncol/1.5, figheight/1.5))
fontsize = 0.2 * 70


ax = fig1.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)

ax.set_xlim(0, 4*ncol)
ax.set_ylim(0, figheight)

def to_texstring(s):
    s = s.replace("<", r"$<$")
    s = s.replace(">", r"$>$")
    s = s.replace("|", r"$|$")
    return s

for i, (stylename, styleclass) in enumerate(sorted(styles.items())):
    x = 3.2 + (i//nrow)*4
    y = (figheight - 0.7 - i%nrow) # /figheight
    p = mpatches.Circle((x, y), 0.2, fc="w")
    ax.add_patch(p)

    ax.annotate(to_texstring(stylename), (x, y),
                (x-1.2, y),
                #xycoords="figure fraction", textcoords="figure fraction",
                ha="right", va="center",
                size=fontsize,
                arrowprops=dict(arrowstyle=stylename,
                                patchB=p,
                                shrinkA=5,
                                shrinkB=5,
                                fc="w", ec="k",
                                connectionstyle="arc3,rad=-0.05",
                                ),
                bbox=dict(boxstyle="square", fc="w"))

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)



plt.draw()
plt.show()




********************************************
./pylab_examples/subplots_demo.py
"""Examples illustrating the use of plt.subplots().

This function creates a figure and a grid of subplots with a single call, while
providing reasonable control over how the individual plots are created.  For
very refined tuning of subplot creation, you can still use add_subplot()
directly on a new figure.
"""

import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.close('all')

# Just a figure and one subplot
f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y)

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing both axes')
ax2.scatter(x, y)
ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

# row and column sharing
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.plot(x, y)
ax1.set_title('Sharing x per column, y per row')
ax2.scatter(x, y)
ax3.scatter(x, 2 * y ** 2 - 1, color='r')
ax4.plot(x, 2 * y ** 2 - 1, color='r')

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(x, y)
axarr[0, 0].set_title('Axis [0,0]')
axarr[0, 1].scatter(x, y)
axarr[0, 1].set_title('Axis [0,1]')
axarr[1, 0].plot(x, y ** 2)
axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 1].scatter(x, y ** 2)
axarr[1, 1].set_title('Axis [1,1]')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

# Four polar axes
plt.subplots(2, 2, subplot_kw=dict(polar=True))

plt.show()




********************************************
./pylab_examples/axis_equal_demo.py
'''This example is only interesting when ran in interactive mode'''

from pylab import *

# Plot circle or radius 3

an = linspace(0,2*pi,100)

subplot(221)
plot( 3*cos(an), 3*sin(an) )
title('not equal, looks like ellipse',fontsize=10)

subplot(222)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
title('equal, looks like circle',fontsize=10)

subplot(223)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
axis([-3,3,-3,3])
title('looks like circle, even after changing limits',fontsize=10)

subplot(224)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
axis([-3,3,-3,3])
plot([0,4],[0,4])
title('still equal after adding line',fontsize=10)

show()









********************************************
./pylab_examples/layer_images.py
#!/usr/bin/env python
"""
Layer images above one another using alpha blending
"""
from __future__ import division
from pylab import *

def func3(x,y):
    return (1- x/2 + x**5 + y**3)*exp(-x**2-y**2)

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = arange(-3.0, 3.0, dx)
y = arange(-3.0, 3.0, dy)
X,Y = meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects


xmin, xmax, ymin, ymax = amin(x), amax(x), amin(y), amax(y)
extent = xmin, xmax, ymin, ymax
fig = plt.figure(frameon=False)

Z1 = array(([0,1]*4 + [1,0]*4)*4); Z1.shape = 8,8  # chessboard
im1 = imshow(Z1, cmap=cm.gray, interpolation='nearest',
             extent=extent)
hold(True)

Z2 = func3(X, Y)

im2 = imshow(Z2, cmap=cm.jet, alpha=.9, interpolation='bilinear',
             extent=extent)
#axis([xmin, xmax, ymin, ymax])

show()






********************************************
./pylab_examples/transoffset.py
#!/usr/bin/env python

'''
This illustrates the use of transforms.offset_copy to
make a transform that positions a drawing element such as
a text string at a specified offset in screen coordinates
(dots or inches) relative to a location given in any
coordinates.

Every Artist--the mpl class from which classes such as
Text and Line are derived--has a transform that can be
set when the Artist is created, such as by the corresponding
pylab command.  By default this is usually the Axes.transData
transform, going from data units to screen dots.  We can
use the offset_copy function to make a modified copy of
this transform, where the modification consists of an
offset.
'''

import pylab as P
from matplotlib.transforms import offset_copy

X = P.arange(7)
Y = X**2

fig = P.figure(figsize=(5,10))
ax = P.subplot(2,1,1)

# If we want the same offset for each text instance,
# we only need to make one transform.  To get the
# transform argument to offset_copy, we need to make the axes
# first; the subplot command above is one way to do this.

transOffset = offset_copy(ax.transData, fig=fig,
                            x = 0.05, y=0.10, units='inches')

for x, y in zip(X, Y):
    P.plot((x,),(y,), 'ro')
    P.text(x, y, '%d, %d' % (int(x),int(y)), transform=transOffset)


# offset_copy works for polar plots also.

ax = P.subplot(2,1,2, polar=True)

transOffset = offset_copy(ax.transData, fig=fig, y = 6, units='dots')

for x, y in zip(X, Y):
    P.polar((x,),(y,), 'ro')
    P.text(x, y, '%d, %d' % (int(x),int(y)),
                transform=transOffset,
                horizontalalignment='center',
                verticalalignment='bottom')


P.show()





********************************************
./pylab_examples/date_index_formatter.py

"""
When plotting daily data, a frequent request is to plot the data
ignoring skips, eg no extra spaces for weekends.  This is particularly
common in financial time series, when you may have data for M-F and
not Sat, Sun and you don't want gaps in the x axis.  The approach is
to simply use the integer index for the xdata and a custom tick
Formatter to get the appropriate date string for a given index.
"""

from __future__ import print_function
import numpy
from matplotlib.mlab import csv2rec
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.ticker import Formatter

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print ('loading %s' % datafile)
r = csv2rec(datafile)[-40:]

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(round(x))
        if ind>=len(self.dates) or ind<0: return ''

        return self.dates[ind].strftime(self.fmt)

formatter = MyFormatter(r.date)

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(formatter)
ax.plot(numpy.arange(len(r)), r.close, 'o-')
fig.autofmt_xdate()
plt.show()




********************************************
./pylab_examples/stackplot_demo.py
import numpy as np
from matplotlib import pyplot as plt

fnx = lambda : np.random.randint(5, 50, 10)
y = np.row_stack((fnx(), fnx(), fnx()))
x = np.arange(10)

y1, y2, y3 = fnx(), fnx(), fnx()

fig, ax = plt.subplots()
ax.stackplot(x, y)
plt.show()

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3)
plt.show()




********************************************
./pylab_examples/nan_test.py
#!/usr/bin/env python
"""
Example: simple line plots with NaNs inserted.
"""
from pylab import *
import numpy as np

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
t[41:60] = NaN
#t[50:60] = np.inf

subplot(2,1,1)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('A sine wave with a gap of NaNs between 0.4 and 0.6')
grid(True)

subplot(2,1,2)
t[0] = NaN
t[-1] = NaN
plot(t, s, '-', lw=2)
title('Also with NaN in first and last point')

xlabel('time (s)')
ylabel('more nans')
grid(True)

show()




********************************************
./pylab_examples/scatter_star_poly.py
import numpy as np
import matplotlib.pyplot as plt


x = np.random.rand(10)
y = np.random.rand(10)
z = np.sqrt(x**2 + y**2)

plt.subplot(321)
plt.scatter(x,y,s=80, c=z, marker=">")

plt.subplot(322)
plt.scatter(x,y,s=80, c=z, marker=(5,0))

verts = list(zip([-1.,1.,1.,-1.],[-1.,-1.,1.,-1.]))
plt.subplot(323)
plt.scatter(x,y,s=80, c=z, marker=(verts,0))
# equivalent:
#plt.scatter(x,y,s=80, c=z, marker=None, verts=verts)

plt.subplot(324)
plt.scatter(x,y,s=80, c=z, marker=(5,1))

plt.subplot(325)
plt.scatter(x,y,s=80, c=z, marker='+')

plt.subplot(326)
plt.scatter(x,y,s=80, c=z, marker=(5,2))

plt.show()




********************************************
./pylab_examples/boxplot_demo2.py
"""
Thanks Josh Hemann for the example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Generate some data from five different probability distributions,
# each with different characteristics. We want to play with how an IID
# bootstrap resample of the data preserves the distributional
# properties of the original sample, and a boxplot is one visual tool
# to make this assessment
numDists = 5
randomDists = ['Normal(1,1)',' Lognormal(1,1)', 'Exp(1)', 'Gumbel(6,4)',
              'Triangular(2,9,11)']
N = 500
norm = np.random.normal(1,1, N)
logn = np.random.lognormal(1,1, N)
expo = np.random.exponential(1, N)
gumb = np.random.gumbel(6, 4, N)
tria = np.random.triangular(2, 9, 11, N)

# Generate some random indices that we'll use to resample the original data
# arrays. For code brevity, just use the same random indices for each array
bootstrapIndices = np.random.random_integers(0, N-1, N)
normBoot = norm[bootstrapIndices]
expoBoot = expo[bootstrapIndices]
gumbBoot = gumb[bootstrapIndices]
lognBoot = logn[bootstrapIndices]
triaBoot = tria[bootstrapIndices]

data = [norm, normBoot,  logn, lognBoot, expo, expoBoot, gumb, gumbBoot,
       tria, triaBoot]

fig, ax1 = plt.subplots(figsize=(10,6))
fig.canvas.set_window_title('A Boxplot Example')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of IID Bootstrap Resampling Across Five Distributions')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')

# Now fill the boxes with desired colors
boxColors = ['darkkhaki','royalblue']
numBoxes = numDists*2
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  # Alternate between Dark Khaki and Royal Blue
  k = i % 2
  boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
  ax1.add_patch(boxPolygon)
  # Now draw the median lines back over what we just filled in
  med = bp['medians'][i]
  medianX = []
  medianY = []
  for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      plt.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
  # Finally, overplot the sample averages, with horizontal alignment
  # in the center of each box
  plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes+0.5)
top = 40
bottom = -5
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=np.repeat(randomDists, 2))
plt.setp(xtickNames, rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes)+1
upperLabels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
   k = tick % 2
   ax1.text(pos[tick], top-(top*0.05), upperLabels[tick],
        horizontalalignment='center', size='x-small', weight=weights[k],
        color=boxColors[k])

# Finally, add a basic legend
plt.figtext(0.80, 0.08,  str(N) + ' Random Numbers' ,
           backgroundcolor=boxColors[0], color='black', weight='roman',
           size='x-small')
plt.figtext(0.80, 0.045, 'IID Bootstrap Resample',
backgroundcolor=boxColors[1],
           color='white', weight='roman', size='x-small')
plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
           weight='roman', size='medium')
plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
           size='x-small')

plt.show()




********************************************
./pylab_examples/finance_demo.py
#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
     DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc


# (Year, month, day) tuples suffice as args for quotes_historical_yahoo
date1 = (2004, 2, 1)
date2 = (2004, 4, 12)


mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12

quotes = quotes_historical_yahoo_ohlc('INTC', date1, date2)
if len(quotes) == 0:
    raise SystemExit

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
#ax.xaxis.set_minor_formatter(dayFormatter)

#plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(ax, quotes, width=0.6)

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()




********************************************
./pylab_examples/spine_placement_demo.py
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
x = np.linspace(-np.pi,np.pi,100)
y = 2*np.sin(x)

ax = fig.add_subplot(2,2,1)
ax.set_title('centered spines')
ax.plot(x,y)
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax = fig.add_subplot(2,2,2)
ax.set_title('zeroed spines')
ax.plot(x,y)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax = fig.add_subplot(2,2,3)
ax.set_title('spines at axes (0.6, 0.1)')
ax.plot(x,y)
ax.spines['left'].set_position(('axes',0.6))
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('axes',0.1))
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax = fig.add_subplot(2,2,4)
ax.set_title('spines at data (1,2)')
ax.plot(x,y)
ax.spines['left'].set_position(('data',1))
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data',2))
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# ----------------------------------------------------

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

fig = plt.figure()

x = np.linspace(0,2*np.pi,100)
y = 2*np.sin(x)

ax = fig.add_subplot(2,2,1)
ax.plot(x,y)
adjust_spines(ax,['left'])

ax = fig.add_subplot(2,2,2)
ax.plot(x,y)
adjust_spines(ax,[])

ax = fig.add_subplot(2,2,3)
ax.plot(x,y)
adjust_spines(ax,['left','bottom'])

ax = fig.add_subplot(2,2,4)
ax.plot(x,y)
adjust_spines(ax,['bottom'])

plt.show()




********************************************
./pylab_examples/movie_demo.py
#!/usr/bin/env python
# -*- noplot -*-

from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np

files = []

fig, ax = plt.subplots(figsize=(5,5))
for i in range(50):  # 50 frames
    plt.cla()
    plt.imshow(np.random.rand(5,5), interpolation='nearest')
    fname = '_tmp%03d.png'%i
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)

print('Making movie animation.mpg - this make take a while')
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
#os.system("convert _tmp*.png animation.mng")

# cleanup
for fname in files: os.remove(fname)




********************************************
./pylab_examples/quiver_demo.py
'''
Demonstration of quiver and quiverkey functions. This is using the
new version coming from the code in quiver.py.

Known problem: the plot autoscaling does not take into account
the arrows, so those on the boundaries are often out of the picture.
This is *not* an easy problem to solve in a perfectly general way.
The workaround is to manually expand the axes.

'''
from pylab import *
from numpy import ma

X,Y = meshgrid( arange(0,2*pi,.2),arange(0,2*pi,.2) )
U = cos(X)
V = sin(Y)

#1
figure()
Q = quiver( U, V)
qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
               fontproperties={'weight': 'bold'})
l,r,b,t = axis()
dx, dy = r-l, t-b
axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])

title('Minimal arguments, no kwargs')

#2
figure()
Q = quiver( X, Y, U, V, units='width')
qk = quiverkey(Q, 0.9, 0.95, 2, r'$2 \frac{m}{s}$',
               labelpos='E',
               coordinates='figure',
               fontproperties={'weight': 'bold'})
axis([-1, 7, -1, 7])
title('scales with plot width, not view')

#3
figure()
Q = quiver( X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
            pivot='mid', color='r', units='inches' )
qk = quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$', fontproperties={'weight': 'bold'})
plot( X[::3, ::3], Y[::3, ::3], 'k.')
axis([-1, 7, -1, 7])
title("pivot='mid'; every third arrow; units='inches'")

#4
figure()
M = sqrt(pow(U, 2) + pow(V, 2))
Q = quiver( X, Y, U, V, M, units='x', pivot='tip', width=0.022, scale=1/0.15)
qk = quiverkey(Q, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                            labelpos='E',
                            fontproperties={'weight': 'bold'})
plot(X, Y, 'k.')
axis([-1, 7, -1, 7])
title("scales with x view; pivot='tip'")

#5
figure()
Q = quiver( X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
             color='r', units='x',
            linewidths=(2,), edgecolors=('k'), headaxislength=5 )
qk = quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$', fontproperties={'weight': 'bold'})
axis([-1, 7, -1, 7])
title("triangular head; scale with x view; black edges")

#6
figure()
M = zeros(U.shape, dtype='bool')
M[U.shape[0]/3:2*U.shape[0]/3,U.shape[1]/3:2*U.shape[1]/3] = True
U = ma.masked_array(U, mask=M)
V = ma.masked_array(V, mask=M)
Q = quiver( U, V)
qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
               fontproperties={'weight': 'bold'})
l,r,b,t = axis()
dx, dy = r-l, t-b
axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])
title('Minimal arguments, no kwargs - masked values')


show()





********************************************
./pylab_examples/webapp_demo.py
#!/usr/bin/env python
# -*- noplot -*-
"""
This example shows how to use the agg backend directly to create
images, which may be of use to web application developers who want
full control over their code without using the pyplot interface to
manage figures, figure closing etc.

.. note::

    It is not necessary to avoid using the pyplot interface in order to
    create figures without a graphical front-end - simply setting
    the backend to "Agg" would be sufficient.


It is also worth noting that, because matplotlib can save figures to file-like
object, matplotlib can also be used inside a cgi-script *without* needing to
write a figure to disk.

"""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def make_fig():
    """
    Make a figure and save it to "webagg.png".

    """
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([1, 2, 3], 'ro--', markersize=12, markerfacecolor='g')

    # make a translucent scatter collection
    x = np.random.rand(100)
    y = np.random.rand(100)
    area = np.pi * (10 * np.random.rand(100)) ** 2 # 0 to 10 point radiuses
    c = ax.scatter(x, y, area)
    c.set_alpha(0.5)

    # add some text decoration
    ax.set_title('My first image')
    ax.set_ylabel('Some numbers')
    ax.set_xticks((.2, .4, .6, .8))
    labels = ax.set_xticklabels(('Bill', 'Fred', 'Ted', 'Ed'))

    # To set object properties, you can either iterate over the
    # objects manually, or define you own set command, as in setapi
    # above.
    for label in labels:
        label.set_rotation(45)
        label.set_fontsize(12)

    FigureCanvasAgg(fig).print_png('webapp.png', dpi=150)

make_fig()



********************************************
./pylab_examples/colours.py
#!/usr/bin/env python
# -*- noplot -*-
"""
Some simple functions to generate colours.
"""
import numpy as np
from matplotlib.colors import colorConverter

def pastel(colour, weight=2.4):
    """ Convert colour into a nice pastel shade"""
    rgb = np.asarray(colorConverter.to_rgb(colour))
    # scale colour
    maxc = max(rgb)
    if maxc < 1.0 and maxc > 0:
        # scale colour
        scale = 1.0 / maxc
        rgb = rgb * scale
    # now decrease saturation
    total = rgb.sum()
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [c + (x * (1.0-c)) for c in rgb]

    return rgb

def get_colours(n):
    """ Return n pastel colours. """
    base = np.asarray([[1,0,0], [0,1,0], [0,0,1]])

    if n <= 3:
        return base[0:n]

    # how many new colours to we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)

    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1, needed[start]+2):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start+1] * x))

    return [pastel(c) for c in colours[0:n]]





********************************************
./pylab_examples/legend_demo_custom_handler.py
import matplotlib.pyplot as plt
import numpy as np

ax = plt.subplot(111, aspect=1)
x, y = np.random.randn(2, 20)
#[1.1, 2, 2.8], [1.1, 2, 1.8]
l1, = ax.plot(x,y, "k+", mew=3, ms=12)
l2, = ax.plot(x,y, "w+", mew=1, ms=10)

import matplotlib.patches as mpatches
c = mpatches.Circle((0, 0), 1, fc="g", ec="r", lw=3)
ax.add_patch(c)



from matplotlib.legend_handler import HandlerPatch

def make_legend_ellipse(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
    p = mpatches.Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                         width = width+xdescent, height=(height+ydescent))

    return p

plt.legend([c, (l1, l2)], ["Label 1", "Label 2"],
           handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_ellipse),
                        })

plt.show()




********************************************
./pylab_examples/findobj_demo.py
"""
Recursively find all objects that match some criteria
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text

a = np.arange(0,3,.02)
b = np.arange(0,3,.02)
c = np.exp(a)
d = c[::-1]

fig, ax = plt.subplots()
plt.plot(a,c,'k--',a,d,'k:',a,c+d,'k')
plt.legend(('Model length', 'Data length', 'Total message length'),
           'upper center', shadow=True)
plt.ylim([-1,20])
plt.grid(False)
plt.xlabel('Model complexity --->')
plt.ylabel('Message length --->')
plt.title('Minimum Message Length')

# match on arbitrary function
def myfunc(x):
    return hasattr(x, 'set_color') and not hasattr(x, 'set_facecolor')

for o in fig.findobj(myfunc):
    o.set_color('blue')

# match on class instances
for o in fig.findobj(text.Text):
    o.set_fontstyle('italic')



plt.show()




********************************************
./pylab_examples/figimage_demo.py
"""
This illustrates placing images directly in the figure, with no axes.

"""
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


fig = plt.figure()
Z = np.arange(10000.0)
Z.shape = 100,100
Z[:,50:] = 1.

im1 = plt.figimage(Z, xo=50, yo=0, cmap=cm.jet, origin='lower')
im2 = plt.figimage(Z, xo=100, yo=100, alpha=.8, cmap=cm.jet, origin='lower')

plt.show()







********************************************
./pylab_examples/tripcolor_demo.py
"""
Pseudocolor plots of unstructured triangular grids.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math

# Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.

# First create the x and y coordinates of the points.
n_angles = 36
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
z = (np.cos(radii)*np.cos(angles*3.0)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

# tripcolor plot.
plt.figure()
plt.gca().set_aspect('equal')
plt.tripcolor(triang, z, shading='flat', cmap=plt.cm.rainbow)
plt.colorbar()
plt.title('tripcolor of Delaunay triangulation, flat shading')

# Illustrate Gouraud shading.
plt.figure()
plt.gca().set_aspect('equal')
plt.tripcolor(triang, z, shading='gouraud', cmap=plt.cm.rainbow)
plt.colorbar()
plt.title('tripcolor of Delaunay triangulation, gouraud shading')


# You can specify your own triangulation rather than perform a Delaunay
# triangulation of the points, where each triangle is given by the indices of
# the three points that make up the triangle, ordered in either a clockwise or
# anticlockwise manner.

xy = np.asarray([
    [-0.101, 0.872], [-0.080, 0.883], [-0.069, 0.888], [-0.054, 0.890],
    [-0.045, 0.897], [-0.057, 0.895], [-0.073, 0.900], [-0.087, 0.898],
    [-0.090, 0.904], [-0.069, 0.907], [-0.069, 0.921], [-0.080, 0.919],
    [-0.073, 0.928], [-0.052, 0.930], [-0.048, 0.942], [-0.062, 0.949],
    [-0.054, 0.958], [-0.069, 0.954], [-0.087, 0.952], [-0.087, 0.959],
    [-0.080, 0.966], [-0.085, 0.973], [-0.087, 0.965], [-0.097, 0.965],
    [-0.097, 0.975], [-0.092, 0.984], [-0.101, 0.980], [-0.108, 0.980],
    [-0.104, 0.987], [-0.102, 0.993], [-0.115, 1.001], [-0.099, 0.996],
    [-0.101, 1.007], [-0.090, 1.010], [-0.087, 1.021], [-0.069, 1.021],
    [-0.052, 1.022], [-0.052, 1.017], [-0.069, 1.010], [-0.064, 1.005],
    [-0.048, 1.005], [-0.031, 1.005], [-0.031, 0.996], [-0.040, 0.987],
    [-0.045, 0.980], [-0.052, 0.975], [-0.040, 0.973], [-0.026, 0.968],
    [-0.020, 0.954], [-0.006, 0.947], [ 0.003, 0.935], [ 0.006, 0.926],
    [ 0.005, 0.921], [ 0.022, 0.923], [ 0.033, 0.912], [ 0.029, 0.905],
    [ 0.017, 0.900], [ 0.012, 0.895], [ 0.027, 0.893], [ 0.019, 0.886],
    [ 0.001, 0.883], [-0.012, 0.884], [-0.029, 0.883], [-0.038, 0.879],
    [-0.057, 0.881], [-0.062, 0.876], [-0.078, 0.876], [-0.087, 0.872],
    [-0.030, 0.907], [-0.007, 0.905], [-0.057, 0.916], [-0.025, 0.933],
    [-0.077, 0.990], [-0.059, 0.993]])
x = xy[:, 0]*180/3.14159
y = xy[:, 1]*180/3.14159

triangles = np.asarray([
    [67, 66,  1], [65,  2, 66], [ 1, 66,  2], [64,  2, 65], [63,  3, 64],
    [60, 59, 57], [ 2, 64,  3], [ 3, 63,  4], [ 0, 67,  1], [62,  4, 63],
    [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [ 4, 62, 68],
    [ 6,  5,  9], [61, 68, 62], [69, 68, 61], [ 9,  5, 70], [ 6,  8,  7],
    [ 4, 70,  5], [ 8,  6,  9], [56, 69, 57], [69, 56, 52], [70, 10,  9],
    [54, 53, 55], [56, 55, 53], [68, 70,  4], [52, 56, 53], [11, 10, 12],
    [69, 71, 68], [68, 13, 70], [10, 70, 13], [51, 50, 52], [13, 68, 71],
    [52, 71, 69], [12, 10, 13], [71, 52, 50], [71, 14, 13], [50, 49, 71],
    [49, 48, 71], [14, 16, 15], [14, 71, 48], [17, 19, 18], [17, 20, 19],
    [48, 16, 14], [48, 47, 16], [47, 46, 16], [16, 46, 45], [23, 22, 24],
    [21, 24, 22], [17, 16, 45], [20, 17, 45], [21, 25, 24], [27, 26, 28],
    [20, 72, 21], [25, 21, 72], [45, 72, 20], [25, 28, 26], [44, 73, 45],
    [72, 45, 73], [28, 25, 29], [29, 25, 31], [43, 73, 44], [73, 43, 40],
    [72, 73, 39], [72, 31, 25], [42, 40, 43], [31, 30, 29], [39, 73, 40],
    [42, 41, 40], [72, 33, 31], [32, 31, 33], [39, 38, 72], [33, 72, 38],
    [33, 38, 34], [37, 35, 38], [34, 38, 35], [35, 37, 36]])

xmid = x[triangles].mean(axis=1)
ymid = y[triangles].mean(axis=1)
x0 = -5
y0 = 52
zfaces = np.exp(-0.01*((xmid-x0)*(xmid-x0) + (ymid-y0)*(ymid-y0)))

# Rather than create a Triangulation object, can simply pass x, y and triangles
# arrays to tripcolor directly.  It would be better to use a Triangulation
# object if the same triangulation was to be used more than once to save
#duplicated calculations.
# Can specify one color value per face rather than one per point by using the
# facecolors kwarg.
plt.figure()
plt.gca().set_aspect('equal')
plt.tripcolor(x, y, triangles, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.title('tripcolor of user-specified triangulation')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')

plt.show()




********************************************
./pylab_examples/data_helper.py
#!/usr/bin/env python
# Some functions to load a return data for the plot demos

from numpy import fromstring, argsort, take, array, resize
import matplotlib.cbook as cbook
def get_two_stock_data():
    """
    load stock time and price data for two stocks The return values
    (d1,p1,d2,p2) are the trade time (in days) and prices for stocks 1
    and 2 (intc and aapl)
    """
    ticker1, ticker2 = 'INTC', 'AAPL'

    file1 = cbook.get_sample_data('INTC.dat.gz')
    file2 = cbook.get_sample_data('AAPL.dat.gz')
    M1 = fromstring( file1.read(), '<d')

    M1 = resize(M1, (M1.shape[0]/2,2) )

    M2 = fromstring( file2.read(), '<d')
    M2 = resize(M2, (M2.shape[0]/2,2) )

    d1, p1 = M1[:,0], M1[:,1]
    d2, p2 = M2[:,0], M2[:,1]
    return (d1,p1,d2,p2)


def get_daily_data():
    """
    return stock1 and stock2 instances, each of which have attributes

      open, high, low, close, volume

    as numeric arrays

    """
    class C: pass

    def get_ticker(ticker):
        vals = []

        datafile = cbook.get_sample_data('%s.csv'%ticker, asfileobj=False)

        lines = open(datafile).readlines()
        for line in lines[1:]:
            vals.append([float(val) for val in line.split(',')[1:]])

        M = array(vals)
        c = C()
        c.open = M[:,0]
        c.high = M[:,1]
        c.low = M[:,2]
        c.close = M[:,3]
        c.volume = M[:,4]
        return c
    c1 = get_ticker('intc')
    c2 = get_ticker('msft')
    return c1, c2




********************************************
./pylab_examples/scatter_hist.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)

nullfmt   = NullFormatter()         # no labels

# definitions for the axes 
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8,8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
lim = ( int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim( (-lim, lim) )
axScatter.set_ylim( (-lim, lim) )

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

axHistx.set_xlim( axScatter.get_xlim() )
axHisty.set_ylim( axScatter.get_ylim() )

plt.show()




********************************************
./pylab_examples/histogram_percent_demo.py
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'

x = randn(5000)

# Make a normed histogram. It'll be multiplied by 100 later.
plt.hist(x, bins=50, normed=True)

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
formatter = FuncFormatter(to_percent)

# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()




********************************************
./pylab_examples/date_demo1.py
#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks

All matplotlib date plotting is done by converting date instances into
days since the 0001-01-01 UTC.  The conversion, tick locating and
formatting is done behind the scenes so this is most transparent to
you.  The dates module provides several converter functions date2num
and num2date

This example requires an active internet connection since it uses
yahoo finance to get the data for plotting
"""

import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
date1 = datetime.date(1995, 1, 1)
date2 = datetime.date(2004, 4, 12)

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

quotes = quotes_historical_yahoo_ochl(
    'INTC', date1, date2)
if len(quotes) == 0:
    raise SystemExit

dates = [q[0] for q in quotes]
opens = [q[1] for q in quotes]

fig, ax = plt.subplots()
ax.plot_date(dates, opens, '-')

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
def price(x): return '$%1.2f'%x
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = price
ax.grid(True)

fig.autofmt_xdate()
plt.show()




********************************************
./pylab_examples/figure_title.py
#!/usr/bin/env python
from matplotlib.font_manager import FontProperties
from pylab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)
t3 = arange(0.0, 2.0, 0.01)


subplot(121)
plot(t1, f(t1), 'bo', t2, f(t2), 'k')
title('subplot 1')
ylabel('Damped oscillation')
suptitle('This is a somewhat long figure title', fontsize=16)


subplot(122)
plot(t3, cos(2*pi*t3), 'r--')
xlabel('time (s)')
title('subplot 2')
ylabel('Undamped')

show()





********************************************
./pylab_examples/multiple_yaxis_with_spines.py
import matplotlib.pyplot as plt

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its 
# detached spine is invisible.  First, activate the frame but make the patch 
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

host.set_xlim(0, 2)
host.set_ylim(0, 2)
par1.set_ylim(0, 4)
par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines])

plt.show()




********************************************
./pylab_examples/anscombe.py
#!/usr/bin/env python

from __future__ import print_function
"""
Edward Tufte uses this example from Anscombe to show 4 datasets of x
and y that have the same mean, standard deviation, and regression
line, but which are qualitatively different.

matplotlib fun for a rainy day
"""

from pylab import *

x =  array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = array([8,8,8,8,8,8,8,19,8,8,8])
y4 = array([6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89])

def fit(x):
    return 3+0.5*x



xfit = array( [amin(x), amax(x) ] )

subplot(221)
plot(x,y1,'ks', xfit, fit(xfit), 'r-', lw=2)
axis([2,20,2,14])
setp(gca(), xticklabels=[], yticks=(4,8,12), xticks=(0,10,20))
text(3,12, 'I', fontsize=20)

subplot(222)
plot(x,y2,'ks', xfit, fit(xfit), 'r-', lw=2)
axis([2,20,2,14])
setp(gca(), xticklabels=[], yticks=(4,8,12), yticklabels=[], xticks=(0,10,20))
text(3,12, 'II', fontsize=20)

subplot(223)
plot(x,y3,'ks', xfit, fit(xfit), 'r-', lw=2)
axis([2,20,2,14])
text(3,12, 'III', fontsize=20)
setp(gca(), yticks=(4,8,12), xticks=(0,10,20))

subplot(224)

xfit = array([amin(x4),amax(x4)])
plot(x4,y4,'ks', xfit, fit(xfit), 'r-', lw=2)
axis([2,20,2,14])
setp(gca(), yticklabels=[], yticks=(4,8,12), xticks=(0,10,20))
text(3,12, 'IV', fontsize=20)

#verify the stats
pairs = (x,y1), (x,y2), (x,y3), (x4,y4)
for x,y in pairs:
    print ('mean=%1.2f, std=%1.2f, r=%1.2f'%(mean(y), std(y), corrcoef(x,y)[0][1]))

show()




********************************************
./pylab_examples/errorbar_subsample.py
'''
Demo for the errorevery keyword to show data full accuracy data plots with
few errorbars.
'''

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.1)
y = np.exp(-x)

# example variable error bar values
yerr = 0.1 + 0.1*np.sqrt(x)


# Now switch to a more OO interface to exercise more features.
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
ax = axs[0]
ax.errorbar(x, y, yerr=yerr)
ax.set_title('all errorbars')

ax = axs[1]
ax.errorbar(x, y, yerr=yerr, errorevery=5)
ax.set_title('only every 5th errorbar')


fig.suptitle('Errorbar subsampling for better visualibility')

plt.show()




********************************************
./pylab_examples/set_and_get.py
"""

MATLAB and pylab allow you to use setp and get to set and get
object properties, as well as to do introspection on the object

set
    To set the linestyle of a line to be dashed, you can do

      >>> line, = plot([1,2,3])
      >>> setp(line, linestyle='--')

    If you want to know the valid types of arguments, you can provide the
    name of the property you want to set without a value

      >>> setp(line, 'linestyle')
          linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]

    If you want to see all the properties that can be set, and their
    possible values, you can do

        >>> setp(line)

    set operates on a single instance or a list of instances.  If you are
    in query mode introspecting the possible values, only the first
    instance in the sequence is used.  When actually setting values, all
    the instances will be set.  e.g., suppose you have a list of two lines,
    the following will make both lines thicker and red

        >>> x = arange(0,1.0,0.01)
        >>> y1 = sin(2*pi*x)
        >>> y2 = sin(4*pi*x)
        >>> lines = plot(x, y1, x, y2)
        >>> setp(lines, linewidth=2, color='r')


get:

    get returns the value of a given attribute.  You can use get to query
    the value of a single attribute

        >>> getp(line, 'linewidth')
            0.5

    or all the attribute/value pairs

    >>> getp(line)
        aa = True
        alpha = 1.0
        antialiased = True
        c = b
        clip_on = True
        color = b
        ... long listing skipped ...

Aliases:

  To reduce keystrokes in interactive mode, a number of properties
  have short aliases, eg 'lw' for 'linewidth' and 'mec' for
  'markeredgecolor'.  When calling set or get in introspection mode,
  these properties will be listed as 'fullname or aliasname', as in




"""

from __future__ import print_function
from pylab import *


x = arange(0,1.0,0.01)
y1 = sin(2*pi*x)
y2 = sin(4*pi*x)
lines = plot(x, y1, x, y2)
l1, l2 = lines
setp(lines, linestyle='--')       # set both to dashed
setp(l1, linewidth=2, color='r')  # line1 is thick and red
setp(l2, linewidth=1, color='g')  # line2 is thicker and green


print ('Line setters')
setp(l1)
print ('Line getters')
getp(l1)

print ('Rectangle setters')
setp(gca().patch)
print ('Rectangle getters')
getp(gca().patch)

t = title('Hi mom')
print ('Text setters')
setp(t)
print ('Text getters')
getp(t)

show()




********************************************
./pylab_examples/triinterp_demo.py
"""
Interpolation from triangular grid to quad grid.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# Create triangulation.
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])
y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0])
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]
triang = mtri.Triangulation(x, y, triangles)

# Interpolate to regularly-spaced quad grid.
z = np.cos(1.5*x)*np.cos(1.5*y)
xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))

interp_lin = mtri.LinearTriInterpolator(triang, z)
zi_lin = interp_lin(xi, yi)

interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
zi_cubic_geom = interp_cubic_geom(xi, yi)

interp_cubic_min_E = mtri.CubicTriInterpolator(triang, z, kind='min_E')
zi_cubic_min_E = interp_cubic_min_E(xi, yi)


# Plot the triangulation.
plt.subplot(221)
plt.tricontourf(triang, z)
plt.triplot(triang, 'ko-')
plt.title('Triangular grid')

# Plot linear interpolation to quad grid.
plt.subplot(222)
plt.contourf(xi, yi, zi_lin)
plt.plot(xi, yi, 'k-', alpha=0.5)
plt.plot(xi.T, yi.T, 'k-', alpha=0.5)
plt.title("Linear interpolation")

# Plot cubic interpolation to quad grid, kind=geom
plt.subplot(223)
plt.contourf(xi, yi, zi_cubic_geom)
plt.plot(xi, yi, 'k-', alpha=0.5)
plt.plot(xi.T, yi.T, 'k-', alpha=0.5)
plt.title("Cubic interpolation,\nkind='geom'")

# Plot cubic interpolation to quad grid, kind=min_E
plt.subplot(224)
plt.contourf(xi, yi, zi_cubic_min_E)
plt.plot(xi, yi, 'k-', alpha=0.5)
plt.plot(xi.T, yi.T, 'k-', alpha=0.5)
plt.title("Cubic interpolation,\nkind='min_E'")

plt.tight_layout()
plt.show()




********************************************
./pylab_examples/coords_demo.py
#!/usr/bin/env python

"""
An example of how to interact with the plotting canvas by connecting
to move and click events
"""
from __future__ import print_function
import sys
from pylab import *

t = arange(0.0, 1.0, 0.01)
s = sin(2*pi*t)
fig, ax = plt.subplots()
ax.plot(t,s)


def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y

    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print ('data coords %f %f' % (event.xdata, event.ydata))

def on_click(event):
    # get the x and y coords, flip y from top to bottom
    x, y = event.x, event.y
    if event.button==1:
        if event.inaxes is not None:
            print ('data coords %f %f' % (event.xdata, event.ydata))

binding_id = connect('motion_notify_event', on_move)
connect('button_press_event', on_click)

if "test_disconnect" in sys.argv:
    print ("disconnecting console coordinate printout...")
    disconnect(binding_id)

show()




********************************************
./pylab_examples/newscalarformatter_demo.py
#!/usr/bin/env python
# Demonstrating the improvements and options of the proposed new ScalarFormatter
from pylab import *
from matplotlib.ticker import OldScalarFormatter

x=frange(0,1,.01)
f=figure(figsize=(6,6))
f.text(0.5,0.975,'The old formatter',horizontalalignment='center',verticalalignment='top')
subplot(221)
plot(x*1e5+1e10,x*1e-10+1e-5)
gca().xaxis.set_major_formatter(OldScalarFormatter())
gca().yaxis.set_major_formatter(OldScalarFormatter())
subplot(222)
plot(x*1e5,x*1e-4)
gca().xaxis.set_major_formatter(OldScalarFormatter())
gca().yaxis.set_major_formatter(OldScalarFormatter())
subplot(223)
plot(-x*1e5-1e10,-x*1e-5-1e-10)
gca().xaxis.set_major_formatter(OldScalarFormatter())
gca().yaxis.set_major_formatter(OldScalarFormatter())
subplot(224)
plot(-x*1e5,-x*1e-4)
gca().xaxis.set_major_formatter(OldScalarFormatter())
gca().yaxis.set_major_formatter(OldScalarFormatter())

x=frange(0,1,.01)
f=figure(figsize=(6,6))
f.text(0.5,0.975,'The new formatter, default settings',horizontalalignment='center',
       verticalalignment='top')
subplot(221)
plot(x*1e5+1e10,x*1e-10+1e-5)
gca().xaxis.set_major_formatter(ScalarFormatter())
gca().yaxis.set_major_formatter(ScalarFormatter())
subplot(222)
plot(x*1e5,x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter())
gca().yaxis.set_major_formatter(ScalarFormatter())
subplot(223)
plot(-x*1e5-1e10,-x*1e-5-1e-10)
gca().xaxis.set_major_formatter(ScalarFormatter())
gca().yaxis.set_major_formatter(ScalarFormatter())
subplot(224)
plot(-x*1e5,-x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter())
gca().yaxis.set_major_formatter(ScalarFormatter())

x=frange(0,1,.01)
f=figure(figsize=(6,6))
f.text(0.5,0.975,'The new formatter, no numerical offset',horizontalalignment='center',
       verticalalignment='top')
subplot(221)
plot(x*1e5+1e10,x*1e-10+1e-5)
gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
subplot(222)
plot(x*1e5,x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
subplot(223)
plot(-x*1e5-1e10,-x*1e-5-1e-10)
gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
subplot(224)
plot(-x*1e5,-x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

x=frange(0,1,.01)
f=figure(figsize=(6,6))
f.text(0.5,0.975,'The new formatter, with mathtext',horizontalalignment='center',
       verticalalignment='top')
subplot(221)
plot(x*1e5+1e10,x*1e-10+1e-5)
gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
subplot(222)
plot(x*1e5,x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
subplot(223)
plot(-x*1e5-1e10,-x*1e-5-1e-10)
gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
subplot(224)
plot(-x*1e5,-x*1e-4)
gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
show()




********************************************
./pylab_examples/contour_label_demo.py
#!/usr/bin/env python
"""
Illustrate some of the more advanced things that one can do with
contour labels.

See also contour_demo.py.
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

##################################################
# Define our surface
##################################################
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)

##################################################
# Make contour labels using creative float classes
# Follows suggestion of Manuel Metz
##################################################
plt.figure()

# Basic contour plot
CS = plt.contour(X, Y, Z)

# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
     def __repr__(self):
         str = '%.1f' % (self.__float__(),)
         if str[-1]=='0':
             return '%.0f' % self.__float__()
         else:
             return '%.1f' % self.__float__()

# Recast levels to new class
CS.levels = [nf(val) for val in CS.levels ]

# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
     fmt = r'%r \%%'
else:
     fmt = '%r %%'
plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

##################################################
# Label contours with arbitrary strings using a
# dictionary
##################################################
plt.figure()

# Basic contour plot
CS = plt.contour(X, Y, Z)

fmt = {}
strs = [ 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh' ]
for l,s in zip( CS.levels, strs ):
    fmt[l] = s

# Label every other level using strings
plt.clabel(CS,CS.levels[::2],inline=True,fmt=fmt,fontsize=10)

# Use a Formatter

plt.figure()

CS = plt.contour(X, Y, 100**Z, locator=plt.LogLocator())
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
plt.clabel(CS, CS.levels, fmt=fmt)
plt.title("$100^Z$")

plt.show()





********************************************
./pylab_examples/pcolor_log.py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.mlab import bivariate_normal


N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

# A low hump with a spike coming out of the top right.
# Needs to have z/colour axis on a log scale so we see both hump and spike.
# linear scale only shows the spike.
Z1 = bivariate_normal(X, Y, 0.1, 0.2, 1.0, 1.0) + 0.1 * bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)

plt.subplot(2,1,1)
plt.pcolor(X, Y, Z1, norm=LogNorm(vmin=Z1.min(), vmax=Z1.max()), cmap='PuBu_r')
plt.colorbar()

plt.subplot(2,1,2)
plt.pcolor(X, Y, Z1, cmap='PuBu_r')
plt.colorbar()


plt.show()





********************************************
./pylab_examples/marker_path.py
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np


star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)


plt.plot(np.arange(10)**2, '--r', marker=cut_star, markersize=15)

plt.show()




********************************************
./pylab_examples/mathtext_demo.py
#!/usr/bin/env python
"""
Use matplotlib's internal LaTeX parser and layout engine.  For true
latex rendering, see the text.usetex option
"""
import numpy as np
from matplotlib.pyplot import figure, show

fig = figure()
fig.subplots_adjust(bottom=0.2)

ax = fig.add_subplot(111, axisbg='y')
ax.plot([1,2,3], 'r')
x = np.arange(0.0, 3.0, 0.1)

ax.grid(True)
ax.set_xlabel(r'$\Delta_i^j$', fontsize=20)
ax.set_ylabel(r'$\Delta_{i+1}^j$', fontsize=20)
tex = r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin(2 \pi f x_i)$'

ax.text(1, 1.6, tex, fontsize=20, va='bottom')

ax.legend([r"$\sqrt{x^2}$"])

ax.set_title(r'$\Delta_i^j \hspace{0.4} \mathrm{versus} \hspace{0.4} \Delta_{i+1}^j$', fontsize=20)


show()





********************************************
./pylab_examples/contour_image.py
#!/usr/bin/env python
'''
Test combinations of contouring, filled contouring, and image plotting.
For contour labelling, see contour_demo.py.

The emphasis in this demo is on showing how to make contours register
correctly on images, and on how to get both of them oriented as
desired.  In particular, note the usage of the "origin" and "extent"
keyword arguments to imshow and contour.
'''
from pylab import *

#Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.
delta = 0.5

extent = (-3,4,-4,3)

x = arange(-3.0, 4.001, delta)
y = arange(-4.0, 3.001, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = (Z1 - Z2) * 10

levels = arange(-2.0, 1.601, 0.4) # Boost the upper limit to avoid truncation
                                  # errors.

norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

figure()


subplot(2,2,1)

cset1 = contourf(X, Y, Z, levels,
                        cmap=cm.get_cmap(cmap, len(levels)-1),
                        norm=norm,
                        )
# It is not necessary, but for the colormap, we need only the
# number of levels minus 1.  To avoid discretization error, use
# either this number or a large number such as the default (256).

#If we want lines as well as filled regions, we need to call
# contour separately; don't try to change the edgecolor or edgewidth
# of the polygons in the collections returned by contourf.
# Use levels output from previous call to guarantee they are the same.
cset2 = contour(X, Y, Z, cset1.levels,
                        colors = 'k',
                        hold='on')
# We don't really need dashed contour lines to indicate negative
# regions, so let's turn them off.
for c in cset2.collections:
    c.set_linestyle('solid')

# It is easier here to make a separate call to contour than
# to set up an array of colors and linewidths.
# We are making a thick green line as a zero contour.
# Specify the zero level as a tuple with only 0 in it.
cset3 = contour(X, Y, Z, (0,),
                colors = 'g',
                linewidths = 2,
                hold='on')
title('Filled contours')
colorbar(cset1)
#hot()


subplot(2,2,2)

imshow(Z, extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='upper', extent=extent)
axis(v)
title("Image, origin 'upper'")

subplot(2,2,3)

imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='lower', extent=extent)
axis(v)
title("Image, origin 'lower'")

subplot(2,2,4)

# We will use the interpolation "nearest" here to show the actual
# image pixels.
# Note that the contour lines don't extend to the edge of the box.
# This is intentional. The Z values are defined at the center of each
# image pixel (each color block on the following subplot), so the
# domain that is contoured does not extend beyond these pixel centers.
im = imshow(Z, interpolation='nearest', extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='image', extent=extent)
axis(v)
ylim = get(gca(), 'ylim')
setp(gca(), ylim=ylim[::-1])
title("Image, origin from rc, reversed y-axis")
colorbar(im)

show()




********************************************
./pylab_examples/legend_demo2.py
# Make a legend for specific lines.
import matplotlib.pyplot as plt
import numpy as np


t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = plt.plot(t2, np.exp(-t2))
l2, l3 = plt.plot(t2, np.sin(2 * np.pi * t2), '--go', t1, np.log(1 + t1), '.')
l4, = plt.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 'rs-.')

plt.legend( (l2, l4), ('oscillatory', 'damped'), 'upper right', shadow=True)
plt.xlabel('time')
plt.ylabel('volts')
plt.title('Damped oscillation')
plt.show()







********************************************
./pylab_examples/psd_demo.py
#!/usr/bin/env python
# python

from pylab import *

dt = 0.01
t = arange(0,10,dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]
s = 0.1*sin(2*pi*t) + cnse

subplot(211)
plot(t,s)
subplot(212)
psd(s, 512, 1/dt)

show()
"""
% compare with MATLAB
dt = 0.01;
t = [0:dt:10];
nse = randn(size(t));
r = exp(-t/0.05);
cnse = conv(nse, r)*dt;
cnse = cnse(1:length(t));
s = 0.1*sin(2*pi*t) + cnse;

subplot(211)
plot(t,s)
subplot(212)
psd(s, 512, 1/dt)

"""




********************************************
./pylab_examples/stem_plot.py
#!/usr/bin/env python
from pylab import *

x = linspace(0.1, 2*pi, 10)
markerline, stemlines, baseline = stem(x, cos(x), '-.')
setp(markerline, 'markerfacecolor', 'b')
setp(baseline, 'color','r', 'linewidth', 2)

show()




********************************************
./pylab_examples/aspect_loglog.py
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(1e1, 1e3)
ax1.set_ylim(1e2, 1e3)
ax1.set_aspect(1)
ax1.set_title("adjustable = box")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_adjustable("datalim")
ax2.plot([1,3, 10], [1, 9, 100], "o-")
ax2.set_xlim(1e-1, 1e2)
ax2.set_ylim(1e-1, 1e3)
ax2.set_aspect(1)
ax2.set_title("adjustable = datalim")

plt.draw()
plt.show()




********************************************
./pylab_examples/broken_axis.py
"""
Broken axis example, where the y-axis will have a portion cut out.
"""
import matplotlib.pylab as plt
import numpy as np


# 30 points between 0 0.2] originally made using np.random.rand(30)*.2
pts = np.array([ 0.015,  0.166,  0.133,  0.159,  0.041,  0.024,  0.195,
    0.039, 0.161,  0.018,  0.143,  0.056,  0.125,  0.096,  0.094, 0.051,
    0.043,  0.021,  0.138,  0.075,  0.109,  0.195,  0.05 , 0.074, 0.079,
    0.155,  0.02 ,  0.01 ,  0.061,  0.008])

# Now let's make two outlier points which are far away from everything. 
pts[[3,14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
f,(ax,ax2) = plt.subplots(2,1,sharex=True)

# plot the same data on both axes
ax.plot(pts)
ax2.plot(pts)

# zoom-in / limit the view to different portions of the data
ax.set_ylim(.78,1.) # outliers only
ax2.set_ylim(0,.22) # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off') # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
ax.plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
ax2.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.show()




********************************************
./pylab_examples/fancybox_demo2.py
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

styles = mpatch.BoxStyle.get_styles()

figheight = (len(styles)+.5)
fig1 = plt.figure(1, (4/1.5, figheight/1.5))
fontsize = 0.3 * 72

for i, (stylename, styleclass) in enumerate(styles.items()):
    fig1.text(0.5, (float(len(styles)) - 0.5 - i)/figheight, stylename,
              ha="center",
              size=fontsize,
              transform=fig1.transFigure,
              bbox=dict(boxstyle=stylename, fc="w", ec="k"))
plt.draw()
plt.show()





********************************************
./pylab_examples/mri_with_eeg.py
#!/usr/bin/env python

"""
This now uses the imshow command instead of pcolor which *is much
faster*
"""
from __future__ import division, print_function

import numpy as np

from matplotlib.pyplot import *
from matplotlib.collections import LineCollection
import matplotlib.cbook as cbook
# I use if 1 to break up the different regions of code visually

if 1:   # load the data
    # data are 256x256 16 bit integers
    dfile = cbook.get_sample_data('s1045.ima.gz')
    im = np.fromstring(dfile.read(), np.uint16).astype(float)
    im.shape = 256, 256

if 1: # plot the MRI in pcolor
    subplot(221)
    imshow(im, cmap=cm.gray)
    axis('off')

if 1:  # plot the histogram of MRI intensity
    subplot(222)
    im = np.ravel(im)
    im = im[np.nonzero(im)] # ignore the background
    im = im/(2.0**15) # normalize
    hist(im, 100)
    xticks([-1, -.5, 0, .5, 1])
    yticks([])
    xlabel('intensity')
    ylabel('MRI density')

if 1:   # plot the EEG
    # load the data

    numSamples, numRows = 800,4
    eegfile = cbook.get_sample_data('eeg.dat', asfileobj=False)
    print ('loading eeg %s' % eegfile)
    data = np.fromstring(open(eegfile, 'rb').read(), float)
    data.shape = numSamples, numRows
    t = 10.0 * np.arange(numSamples, dtype=float)/numSamples
    ticklocs = []
    ax = subplot(212)
    xlim(0,10)
    xticks(np.arange(10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin)*0.7 # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax
    ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:,np.newaxis], data[:,i,np.newaxis])))
        ticklocs.append(i*dr)

    offsets = np.zeros((numRows,2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None,
                           )

    ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])

    xlabel('time (s)')

show()




********************************************
./pylab_examples/date_demo_convert.py
#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange

date1 = datetime.datetime( 2000, 3, 2)
date2 = datetime.datetime( 2000, 3, 6)
delta = datetime.timedelta(hours=6)
dates = drange(date1, date2, delta)

y = arange( len(dates)*1.0)

fig, ax = plt.subplots()
ax.plot_date(dates, y*y)

# this is superfluous, since the autoscaler should get it right, but
# use date2num and num2date to convert between dates and floats if
# you want; both date2num and num2date convert an instance or sequence
ax.set_xlim( dates[0], dates[-1] )

# The hour locator takes the hour or sequence of hours you want to
# tick, not the base multiple

ax.xaxis.set_major_locator( DayLocator() )
ax.xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
ax.xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )

ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
fig.autofmt_xdate()

plt.show()




********************************************
./pylab_examples/griddata_demo.py
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
# make up data.
#npts = int(raw_input('enter # of random points to plot:'))
seed(0)
npts = 200
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)
# define grid.
xi = np.linspace(-2.1,2.1,100)
yi = np.linspace(-2.1,2.1,200)
# grid the data.
zi = griddata(x,y,z,xi,yi,interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow, 
                  vmax=abs(zi).max(), vmin=-abs(zi).max())
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5,zorder=10)
plt.xlim(-2,2)    
plt.ylim(-2,2)     
plt.title('griddata test (%d points)' % npts)
plt.show()

# test case that scikits.delaunay fails on, but natgrid passes..
#data = np.array([[-1, -1], [-1, 0], [-1, 1],
#                 [ 0, -1], [ 0, 0], [ 0, 1],
#                 [ 1, -1 - np.finfo(np.float_).eps], [ 1, 0], [ 1, 1],
#                ])
#x = data[:,0]
#y = data[:,1]
#z = x*np.exp(-x**2-y**2)
## define grid.
#xi = np.linspace(-1.1,1.1,100)
#yi = np.linspace(-1.1,1.1,100)
## grid the data.
#zi = griddata(x,y,z,xi,yi)




********************************************
./pylab_examples/arrow_simple_demo.py
import matplotlib.pyplot as plt

ax = plt.axes()
ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()




********************************************
./pylab_examples/scatter_symbol.py
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

x = np.arange(0.0, 50.0, 2.0)
y = x ** 1.3 + np.random.rand(*x.shape) * 30.0
s = np.random.rand(*x.shape) * 800 + 500

plt.scatter(x, y, s, c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="Luck")
plt.xlabel("Leprechauns")
plt.ylabel("Gold")
plt.legend(loc=2)
plt.show()




********************************************
./pylab_examples/multicolored_line.py
#!/usr/bin/env python
'''
Color parts of a line based on its properties, e.g., slope.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# Create a colormap for red, green and blue and a norm to color
# f' < -0.5 red, f' > 0.5 blue, and the rest green
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the line collection object, setting the colormapping parameters.
# Have to set the actual values used for colormapping separately.
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(z)
lc.set_linewidth(3)

fig1 = plt.figure()
plt.gca().add_collection(lc)
plt.xlim(x.min(), x.max())
plt.ylim(-1.1, 1.1)

# Now do a second plot coloring the curve using a continuous colormap
t = np.linspace(0, 10, 200)
x = np.cos(np.pi * t)
y = np.sin(t)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=plt.get_cmap('copper'),
    norm=plt.Normalize(0, 10))
lc.set_array(t)
lc.set_linewidth(3)

fig2 = plt.figure()
plt.gca().add_collection(lc)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()




********************************************
./pylab_examples/custom_figure_class.py
"""
You can pass a custom Figure constructor to figure if you want to derive from the default Figure.  This simple example creates a figure with a figure title
"""
from matplotlib.pyplot import figure, show
from matplotlib.figure import Figure

class MyFigure(Figure):
    def __init__(self, *args, **kwargs):
        """
        custom kwarg figtitle is a figure title
        """
        figtitle = kwargs.pop('figtitle', 'hi mom')
        Figure.__init__(self, *args, **kwargs)
        self.text(0.5, 0.95, figtitle, ha='center')

fig = figure(FigureClass=MyFigure, figtitle='my title')
ax = fig.add_subplot(111)
ax.plot([1,2,3])

show()





********************************************
./pylab_examples/ellipse_demo.py
from pylab import figure, show, rand
from matplotlib.patches import Ellipse

NUM = 250

ells = [Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360)
        for i in range(NUM)]

fig = figure()
ax = fig.add_subplot(111, aspect='equal')
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(rand())
    e.set_facecolor(rand(3))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

show()




********************************************
./pylab_examples/equal_aspect_ratio.py
#!/usr/bin/env python
"""
Example: simple line plot.
Show how to make a plot that has equal aspect ratio
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)

axes().set_aspect('equal', 'datalim')


show()




********************************************
./pylab_examples/image_nonuniform.py
'''
This illustrates the NonUniformImage class, which still needs
an axes method interface; either a separate interface, or a
generalization of imshow.
'''

from matplotlib.pyplot import figure, show
import numpy as np
from matplotlib.image import NonUniformImage
from matplotlib import cm

interp='nearest'

x = np.linspace(-4, 4, 9)
x2 = x**3
y = np.linspace(-4, 4, 9)
#print 'Size %d points' % (len(x) * len(y))
z = np.sqrt(x[np.newaxis,:]**2 + y[:,np.newaxis]**2)

fig = figure()
fig.suptitle('NonUniformImage class')
ax = fig.add_subplot(221)
im = NonUniformImage(ax, interpolation=interp, extent=(-4,4,-4,4), 
                     cmap=cm.Purples)
im.set_data(x, y, z)
ax.images.append(im)
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_title(interp)

ax = fig.add_subplot(222)
im = NonUniformImage(ax, interpolation=interp, extent=(-64,64,-4,4), 
                     cmap=cm.Purples)
im.set_data(x2, y, z)
ax.images.append(im)
ax.set_xlim(-64,64)
ax.set_ylim(-4,4)
ax.set_title(interp)

interp = 'bilinear'

ax = fig.add_subplot(223)
im = NonUniformImage(ax, interpolation=interp, extent=(-4,4,-4,4), 
                     cmap=cm.Purples)
im.set_data(x, y, z)
ax.images.append(im)
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_title(interp)

ax = fig.add_subplot(224)
im = NonUniformImage(ax, interpolation=interp, extent=(-64,64,-4,4), 
                     cmap=cm.Purples)
im.set_data(x2, y, z)
ax.images.append(im)
ax.set_xlim(-64,64)
ax.set_ylim(-4,4)
ax.set_title(interp)

show()




********************************************
./pylab_examples/fill_betweenx_demo.py
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
import numpy as np

## Copy of fill_between.py but using fill_betweenx() instead.

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)

fig = figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)

ax1.fill_betweenx(x, 0, y1)
ax1.set_ylabel('between y1 and 0')

ax2.fill_betweenx(x, y1, 1)
ax2.set_ylabel('between y1 and 1')

ax3.fill_betweenx(x, y1, y2)
ax3.set_ylabel('between y1 and y2')
ax3.set_xlabel('x')

# now fill between y1 and y2 where a logical condition is met.  Note
# this is different than calling
#   fill_between(x[where], y1[where],y2[where]
# because of edge effects over multiple contiguous regions.
fig = figure()
ax = fig.add_subplot(211)
ax.plot(y1, x, y2, x, color='black')
ax.fill_betweenx(x, y1, y2, where=y2>=y1, facecolor='green')
ax.fill_betweenx(x, y1, y2, where=y2<=y1, facecolor='red')
ax.set_title('fill between where')

# Test support for masked arrays.
y2 = np.ma.masked_greater(y2, 1.0)
ax1 = fig.add_subplot(212, sharex=ax)
ax1.plot(y1, x, y2, x, color='black')
ax1.fill_betweenx(x, y1, y2, where=y2>=y1, facecolor='green')
ax1.fill_betweenx(x, y1, y2, where=y2<=y1, facecolor='red')
ax1.set_title('Now regions with y2 > 1 are masked')

# This example illustrates a problem; because of the data
# gridding, there are undesired unfilled triangles at the crossover
# points.  A brute-force solution would be to interpolate all
# arrays to a very fine grid before plotting.

show()




********************************************
./pylab_examples/load_converter.py
from __future__ import print_function
from matplotlib.dates import strpdate2num
#from matplotlib.mlab import load
import numpy as np
from pylab import figure, show
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)

dates, closes = np.loadtxt(
    datafile, delimiter=',',
    converters={0:strpdate2num('%d-%b-%y')},
    skiprows=1, usecols=(0,2), unpack=True)

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, closes, '-')
fig.autofmt_xdate()
show()




********************************************
./pylab_examples/zorder_demo.py
#!/usr/bin/env python
"""
The default drawing order for axes is patches, lines, text.  This
order is determined by the zorder attribute.  The following defaults
are set

Artist                      Z-order
Patch / PatchCollection      1
Line2D / LineCollection      2
Text                         3

You can change the order for individual artists by setting the zorder.  Any
individual plot() call can set a value for the zorder of that particular item.

In the fist subplot below, the lines are drawn above the patch
collection from the scatter, which is the default.

In the subplot below, the order is reversed.

The second figure shows how to control the zorder of individual lines.
"""

from pylab import *
x = rand(20); y = rand(20)

subplot(211)
plot(x, y, 'r', lw=3)
scatter(x,y,s=120)

subplot(212)
plot(x, y, 'r', zorder=1, lw=3)
scatter(x,y,s=120, zorder=2)

# A new figure, with individually ordered items
x=frange(0,2*pi,npts=100)
figure()
plot(x,sin(x),linewidth=10, color='black',label='zorder=10',zorder = 10)  # on top
plot(x,cos(1.3*x),linewidth=10, color='red', label='zorder=1',zorder = 1) # bottom
plot(x,sin(2.1*x),linewidth=10, color='green', label='zorder=3',zorder = 3)
axhline(0,linewidth=10, color='blue', label='zorder=2',zorder = 2)
l = legend()
l.set_zorder(20) # put the legend on top

show()




********************************************
./pylab_examples/text_rotation_relative_to_line.py
#!/usr/bin/env python
"""
Text objects in matplotlib are normally rotated with respect to the
screen coordinate system (i.e., 45 degrees rotation plots text along a
line that is in between horizontal and vertical no matter how the axes
are changed).  However, at times one wants to rotate text with respect
to something on the plot.  In this case, the correct angle won't be
the angle of that object in the plot coordinate system, but the angle
that that object APPEARS in the screen coordinate system.  This angle
is found by transforming the angle from the plot to the screen
coordinate system, as shown in the example below.
"""
from pylab import *

# Plot diagonal line (45 degrees)
h = plot( r_[:10], r_[:10] )

# set limits so that it no longer looks on screen to be 45 degrees
xlim([-10,20])

# Locations to plot text
l1 = array((1,1))
l2 = array((5,5))

# Rotate angle
angle = 45
trans_angle = gca().transData.transform_angles(array((45,)),
                                               l2.reshape((1,2)))[0]

# Plot text
th1 = text(l1[0],l1[1],'text not rotated correctly',fontsize=16,
           rotation=angle)
th2 = text(l2[0],l2[1],'text not rotated correctly',fontsize=16,
           rotation=trans_angle)

show()




********************************************
./pylab_examples/polar_demo.py
"""
Demo of a line plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


r = np.arange(0, 3.0, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, polar=True)
ax.plot(theta, r, color='r', linewidth=3)
ax.set_rmax(2.0)
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()




********************************************
./pylab_examples/contourf_hatching.py
import matplotlib.pyplot as plt
import numpy as np


# invent some numbers, turning the x and y arrays into simple
# 2d arrays, which make combining them together easier.
x = np.linspace(-3, 5, 150).reshape(1, -1)
y = np.linspace(-3, 5, 120).reshape(-1, 1)
z = np.cos(x) + np.sin(y)

# we no longer need x and y to be 2 dimensional, so flatten them.
x, y = x.flatten(), y.flatten()


# ---------------------------------------------
# |                 Plot #1                   |
# ---------------------------------------------
# the simplest hatched plot with a colorbar
fig = plt.figure()
cs = plt.contourf(x, y, z, hatches=['-', '/', '\\', '//'],
                  cmap=plt.get_cmap('gray'),
                  extend='both', alpha=0.5
                  )
plt.colorbar()


# ---------------------------------------------
# |                 Plot #2                   |
# ---------------------------------------------
# a plot of hatches without color with a legend
plt.figure()
n_levels = 6
plt.contour(x, y, z, n_levels, colors='black', linestyles='-')
cs = plt.contourf(x, y, z, n_levels, colors='none',
                  hatches=['.', '/', '\\', None, '\\\\', '*'],
                  extend='lower'
                  )

# create a legend for the contour set
artists, labels = cs.legend_elements()
plt.legend(artists, labels, handleheight=2)



plt.show()




********************************************
./pylab_examples/image_demo2.py
#!/usr/bin/env python

from __future__ import print_function
from pylab import *
import matplotlib.cbook as cbook

w, h = 512, 512

datafile = cbook.get_sample_data('ct.raw.gz', asfileobj=True)
s = datafile.read()
A = fromstring(s, uint16).astype(float)
A *= 1.0/max(A)
A.shape = w, h

extent = (0, 25, 0, 25)
im = imshow(A, cmap=cm.hot, origin='upper', extent=extent)

markers = [(15.9, 14.5), (16.8, 15)]
x,y = zip(*markers)
plot(x, y, 'o')
#axis([0,25,0,25])



#axis('off')
title('CT density')

if 0:
    x = asum(A,0)
    subplot(212)
    bar(arange(w), x)
    xlim(0,h-1)
    ylabel('density')
    setp(gca(), 'xticklabels', [])

show()




********************************************
./pylab_examples/tricontour_smooth_delaunay.py
"""
Demonstrates high-resolution tricontouring of a random set of points ;
a matplotlib.tri.TriAnalyzer is used to improve the plot quality.

The initial data points and triangular grid for this demo are:
    - a set of random points is instantiated, inside [-1, 1] x [-1, 1] square
    - A Delaunay triangulation of these points is then computed, of which a
    random subset of triangles is masked out by the user (based on
    *init_mask_frac* parameter). This simulates invalidated data.

The proposed generic procedure to obtain a high resolution contouring of such
a data set is the following:
    1) Compute an extended mask with a matplotlib.tri.TriAnalyzer, which will
    exclude badly shaped (flat) triangles from the border of the
    triangulation. Apply the mask to the triangulation (using set_mask).
    2) Refine and interpolate the data using a
    matplotlib.tri.UniformTriRefiner.
    3) Plot the refined data with tricontour.

"""
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


#-----------------------------------------------------------------------------
# Analytical test function
#-----------------------------------------------------------------------------
def experiment_res(x, y):
    """ An analytic function representing experiment results """
    x = 2.*x
    r1 = np.sqrt((0.5-x)**2 + (0.5-y)**2)
    theta1 = np.arctan2(0.5-x, 0.5-y)
    r2 = np.sqrt((-x-0.2)**2 + (-y-0.2)**2)
    theta2 = np.arctan2(-x-0.2, -y-0.2)
    z = (4*(np.exp((r1/10)**2)-1)*30. * np.cos(3*theta1) +
         (np.exp((r2/10)**2)-1)*30. * np.cos(5*theta2) +
         2*(x**2 + y**2))
    return (np.max(z)-z)/(np.max(z)-np.min(z))

#-----------------------------------------------------------------------------
# Generating the initial data test points and triangulation for the demo
#-----------------------------------------------------------------------------
# User parameters for data test points
n_test = 200  # Number of test data points, tested from 3 to 5000 for subdiv=3

subdiv = 3  # Number of recursive subdivisions of the initial mesh for smooth
            # plots. Values >3 might result in a very high number of triangles
            # for the refine mesh: new triangles numbering = (4**subdiv)*ntri

init_mask_frac = 0.0    # Float > 0. adjusting the proportion of
                        # (invalid) initial triangles which will be masked
                        # out. Enter 0 for no mask.

min_circle_ratio = .01  # Minimum circle ratio - border triangles with circle
                        # ratio below this will be masked if they touch a
                        # border. Suggested value 0.01 ; Use -1 to keep
                        # all triangles.

# Random points
random_gen = np.random.mtrand.RandomState(seed=127260)
x_test = random_gen.uniform(-1., 1., size=n_test)
y_test = random_gen.uniform(-1., 1., size=n_test)
z_test = experiment_res(x_test, y_test)

# meshing with Delaunay triangulation
tri = Triangulation(x_test, y_test)
ntri = tri.triangles.shape[0]

# Some invalid data are masked out
mask_init = np.zeros(ntri, dtype=np.bool)
masked_tri = random_gen.randint(0, ntri, int(ntri*init_mask_frac))
mask_init[masked_tri] = True
tri.set_mask(mask_init)


#-----------------------------------------------------------------------------
# Improving the triangulation before high-res plots: removing flat triangles
#-----------------------------------------------------------------------------
# masking badly shaped triangles at the border of the triangular mesh.
mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
tri.set_mask(mask)

# refining the data
refiner = UniformTriRefiner(tri)
tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)

# analytical 'results' for comparison
z_expected = experiment_res(tri_refi.x, tri_refi.y)

# for the demo: loading the 'flat' triangles for plot
flat_tri = Triangulation(x_test, y_test)
flat_tri.set_mask(~mask)


#-----------------------------------------------------------------------------
# Now the plots
#-----------------------------------------------------------------------------
# User options for plots
plot_tri = True          # plot of base triangulation
plot_masked_tri = True   # plot of excessively flat excluded triangles
plot_refi_tri = False    # plot of refined triangulation
plot_expected = False    # plot of analytical function values for comparison


# Graphical options for tricontouring
levels = np.arange(0., 1., 0.025)
cmap = cm.get_cmap(name='Blues', lut=None)

plt.figure()
plt.gca().set_aspect('equal')
plt.title("Filtering a Delaunay mesh\n" +
          "(application to high-resolution tricontouring)")

# 1) plot of the refined (computed) data countours:
plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
               linewidths=[2.0, 0.5, 1.0, 0.5])
# 2) plot of the expected (analytical) data countours (dashed):
if plot_expected:
    plt.tricontour(tri_refi, z_expected, levels=levels, cmap=cmap,
                   linestyles='--')
# 3) plot of the fine mesh on which interpolation was done:
if plot_refi_tri:
    plt.triplot(tri_refi, color='0.97')
# 4) plot of the initial 'coarse' mesh:
if plot_tri:
    plt.triplot(tri, color='0.7')
# 4) plot of the unvalidated triangles from naive Delaunay Triangulation:
if plot_masked_tri:
    plt.triplot(flat_tri, color='red')

plt.show()




********************************************
./pylab_examples/demo_bboximage.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

if __name__ == "__main__":

    fig = plt.figure(1)
    ax = plt.subplot(121)

    txt = ax.text(0.5, 0.5, "test", size=30, ha="center", color="w")
    kwargs = dict()

    bbox_image = BboxImage(txt.get_window_extent,
                           norm = None,
                           origin=None,
                           clip_on=False,
                           **kwargs
                           )
    a = np.arange(256).reshape(1,256)/256.
    bbox_image.set_data(a)
    ax.add_artist(bbox_image)


    ax = plt.subplot(122)
    a = np.linspace(0, 1, 256).reshape(1,-1)
    a = np.vstack((a,a))

    maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
    #nmaps = len(maps) + 1

    #fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)

    ncol = 2
    nrow = len(maps)//ncol + 1

    xpad_fraction = 0.3
    dx = 1./(ncol + xpad_fraction*(ncol-1))

    ypad_fraction = 0.3
    dy = 1./(nrow + ypad_fraction*(nrow-1))

    for i,m in enumerate(maps):
        ix, iy = divmod(i, nrow)
        #plt.figimage(a, 10, i*10, cmap=plt.get_cmap(m), origin='lower')
        bbox0 = Bbox.from_bounds(ix*dx*(1+xpad_fraction),
                                 1.-iy*dy*(1+ypad_fraction)-dy,
                                 dx, dy)
        bbox = TransformedBbox(bbox0, ax.transAxes)

        bbox_image = BboxImage(bbox,
                               cmap = plt.get_cmap(m),
                               norm = None,
                               origin=None,
                               **kwargs
                               )

        bbox_image.set_data(a)
        ax.add_artist(bbox_image)

    plt.draw()
    plt.show()




********************************************
./pylab_examples/barchart_demo2.py
"""
Thanks Josh Hemann for the example

This examples comes from an application in which grade school gym
teachers wanted to be able to show parents how their child did across
a handful of fitness tests, and importantly, relative to how other
children did. To extract the plotting code for demo purposes, we'll
just make up some data for little Johnny Doe...

"""
import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.ticker import MaxNLocator

student = 'Johnny Doe'
grade = 2
gender = 'boy'
cohortSize = 62  # The number of other 2nd grade boys

numTests = 5
testNames = ['Pacer Test', 'Flexed Arm\n Hang', 'Mile Run', 'Agility',
             'Push Ups']
testMeta = ['laps', 'sec', 'min:sec', 'sec', '']
scores = ['7', '48', '12:52', '17', '14']
rankings = np.round(np.random.uniform(0, 1, numTests)*100, 0)


fig, ax1 = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(left=0.115, right=0.88)
fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')
pos = np.arange(numTests)+0.5    # Center bars on the Y-axis ticks
rects = ax1.barh(pos, rankings, align='center', height=0.5, color='m')

ax1.axis([0, 100, 0, 5])
pylab.yticks(pos, testNames)
ax1.set_title('Johnny Doe')
plt.text(50, -0.5, 'Cohort Size: ' + str(cohortSize),
         horizontalalignment='center', size='small')

# Set the right-hand Y-axis ticks and labels and set X-axis tick marks at the
# deciles
ax2 = ax1.twinx()
ax2.plot([100, 100], [0, 5], 'white', alpha=0.1)
ax2.xaxis.set_major_locator(MaxNLocator(11))
xticks = pylab.setp(ax2, xticklabels=['0', '10', '20', '30', '40', '50', '60',
                                      '70', '80', '90', '100'])
ax2.xaxis.grid(True, linestyle='--', which='major', color='grey',
alpha=0.25)
#Plot a solid vertical gridline to highlight the median position
plt.plot([50, 50], [0, 5], 'grey', alpha=0.25)

# Build up the score labels for the right Y-axis by first appending a carriage
# return to each string and then tacking on the appropriate meta information
# (i.e., 'laps' vs 'seconds'). We want the labels centered on the ticks, so if
# there is no meta info (like for pushups) then don't add the carriage return to
# the string


def withnew(i, scr):
    if testMeta[i] != '':
        return '%s\n' % scr
    else:
        return scr

scoreLabels = [withnew(i, scr) for i, scr in enumerate(scores)]
scoreLabels = [i+j for i, j in zip(scoreLabels, testMeta)]
# set the tick locations
ax2.set_yticks(pos)
# set the tick labels
ax2.set_yticklabels(scoreLabels)
# make sure that the limits are set equally on both yaxis so the ticks line up
ax2.set_ylim(ax1.get_ylim())


ax2.set_ylabel('Test Scores')
#Make list of numerical suffixes corresponding to position in a list
#            0     1     2     3     4     5     6     7     8     9
suffixes = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
ax2.set_xlabel('Percentile Ranking Across ' + str(grade) + suffixes[grade]
              + ' Grade ' + gender.title() + 's')

# Lastly, write in the ranking inside each bar to aid in interpretation
for rect in rects:
    # Rectangle widths are already integer-valued but are floating
    # type, so it helps to remove the trailing decimal point and 0 by
    # converting width to int type
    width = int(rect.get_width())

    # Figure out what the last digit (width modulo 10) so we can add
    # the appropriate numerical suffix (e.g., 1st, 2nd, 3rd, etc)
    lastDigit = width % 10
    # Note that 11, 12, and 13 are special cases
    if (width == 11) or (width == 12) or (width == 13):
        suffix = 'th'
    else:
        suffix = suffixes[lastDigit]

    rankStr = str(width) + suffix
    if (width < 5):        # The bars aren't wide enough to print the ranking inside
        xloc = width + 1   # Shift the text to the right side of the right edge
        clr = 'black'      # Black against white background
        align = 'left'
    else:
        xloc = 0.98*width  # Shift the text to the left side of the right edge
        clr = 'white'      # White on magenta
        align = 'right'

    # Center the text vertically in the bar
    yloc = rect.get_y()+rect.get_height()/2.0
    ax1.text(xloc, yloc, rankStr, horizontalalignment=align,
            verticalalignment='center', color=clr, weight='bold')

plt.show()




********************************************
./pylab_examples/bar_stacked.py
#!/usr/bin/env python
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


N = 5
menMeans   = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd     = (2, 3, 4, 1, 2)
womenStd   = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans,   width, color='r', yerr=womenStd)
p2 = plt.bar(ind, womenMeans, width, color='y',
             bottom=menMeans, yerr=menStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind+width/2., ('G1', 'G2', 'G3', 'G4', 'G5') )
plt.yticks(np.arange(0,81,10))
plt.legend( (p1[0], p2[0]), ('Men', 'Women') )

plt.show()




********************************************
./pylab_examples/alignment_test.py
#!/usr/bin/env python
"""
You can precisely layout text in data or axes (0,1) coordinates.  This
example shows you some of the alignment and rotation specifications to
layout text
"""

from pylab import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ax = gca()
p = Rectangle((left, bottom), width, height,
              fill=False,
              )
p.set_transform(ax.transAxes)
p.set_clip_on(False)
ax.add_patch(p)


ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

axis('off')

show()




********************************************
./pylab_examples/psd_demo_complex.py
#This is a ported version of a MATLAB example from the signal processing
#toolbox that showed some difference at one time between Matplotlib's and
#MATLAB's scaling of the PSD.  This differs from psd_demo3.py in that
#this uses a complex signal, so we can see that complex PSD's work properly
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

fs = 1000
t = np.linspace(0, 0.3, 301)
A = np.array([2, 8]).reshape(-1, 1)
f = np.array([150, 140]).reshape(-1, 1)
xn = (A * np.exp(2j * np.pi * f * t)).sum(axis=0) + 5 * np.random.randn(*t.shape)

yticks = np.arange(-50, 30, 10)
xticks = np.arange(-500,550,100)
plt.subplots_adjust(hspace=0.45, wspace=0.3)
ax = plt.subplot(1, 2, 1)

plt.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
    scale_by_freq=True)
plt.title('Periodogram')
plt.yticks(yticks)
plt.xticks(xticks)
plt.grid(True)
plt.xlim(-500, 500)

plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
plt.psd(xn, NFFT=150, Fs=fs, window=mlab.window_none, noverlap=75, pad_to=512,
    scale_by_freq=True)
plt.title('Welch')
plt.xticks(xticks)
plt.yticks(yticks)
plt.ylabel('')
plt.grid(True)
plt.xlim(-500, 500)

plt.show()




********************************************
./pylab_examples/clippedline.py
"""
Clip a line according to the current xlimits, and change the marker
style when zoomed in.

It is not clear this example is still needed or valid; clipping
is now automatic for Line2D objects when x is sorted in
ascending order.

"""

from matplotlib.lines import Line2D
import numpy as np
from pylab import figure, show

class ClippedLine(Line2D):
    """
    Clip the xlimits to the axes view limits -- this example assumes x is sorted
    """

    def __init__(self, ax, *args, **kwargs):
        Line2D.__init__(self, *args, **kwargs)
        self.ax = ax


    def set_data(self, *args, **kwargs):
        Line2D.set_data(self, *args, **kwargs)
        self.recache()
        self.xorig = np.array(self._x)
        self.yorig = np.array(self._y)

    def draw(self, renderer):
        xlim = self.ax.get_xlim()

        ind0, ind1 = np.searchsorted(self.xorig, xlim)
        self._x = self.xorig[ind0:ind1]
        self._y = self.yorig[ind0:ind1]
        N = len(self._x)
        if N<1000:
            self._marker = 's'
            self._linestyle = '-'
        else:
            self._marker = None
            self._linestyle = '-'


        Line2D.draw(self, renderer)


fig = figure()
ax = fig.add_subplot(111, autoscale_on=False)

t = np.arange(0.0, 100.0, 0.01)
s = np.sin(2*np.pi*t)
line = ClippedLine(ax, t, s, color='g', ls='-', lw=2)
ax.add_line(line)
ax.set_xlim(10,30)
ax.set_ylim(-1.1,1.1)
show()






********************************************
./pylab_examples/arrow_demo.py
#!/usr/bin/env python
"""Arrow drawing example for the new fancy_arrow facilities.

Code contributed by: Rob Knight <rob@spot.colorado.edu>

usage:

  python arrow_demo.py realistic|full|sample|extreme


"""
from pylab import *

rates_to_bases={'r1':'AT', 'r2':'TA', 'r3':'GA','r4':'AG','r5':'CA','r6':'AC', \
            'r7':'GT', 'r8':'TG', 'r9':'CT','r10':'TC','r11':'GC','r12':'CG'}
numbered_bases_to_rates = dict([(v,k) for k, v in rates_to_bases.items()])
lettered_bases_to_rates = dict([(v, 'r'+v) for k, v in rates_to_bases.items()])
def add_dicts(d1, d2):
    """Adds two dicts and returns the result."""
    result = d1.copy()
    result.update(d2)
    return result

def make_arrow_plot(data, size=4, display='length', shape='right', \
        max_arrow_width=0.03, arrow_sep = 0.02, alpha=0.5, \
        normalize_data=False, ec=None, labelcolor=None, \
        head_starts_at_zero=True, rate_labels=lettered_bases_to_rates,\
        **kwargs):
    """Makes an arrow plot.

    Parameters:

    data: dict with probabilities for the bases and pair transitions.
    size: size of the graph in inches.
    display: 'length', 'width', or 'alpha' for arrow property to change.
    shape: 'full', 'left', or 'right' for full or half arrows.
    max_arrow_width: maximum width of an arrow, data coordinates.
    arrow_sep: separation between arrows in a pair, data coordinates.
    alpha: maximum opacity of arrows, default 0.8.

    **kwargs can be anything allowed by a Arrow object, e.g.
    linewidth and edgecolor.
    """

    xlim(-0.5,1.5)
    ylim(-0.5,1.5)
    gcf().set_size_inches(size,size)
    xticks([])
    yticks([])
    max_text_size = size*12
    min_text_size = size
    label_text_size = size*2.5
    text_params={'ha':'center', 'va':'center', 'family':'sans-serif',\
        'fontweight':'bold'}
    r2 = sqrt(2)

    deltas = {\
        'AT':(1,0),
        'TA':(-1,0),
        'GA':(0,1),
        'AG':(0,-1),
        'CA':(-1/r2, 1/r2),
        'AC':(1/r2, -1/r2),
        'GT':(1/r2, 1/r2),
        'TG':(-1/r2,-1/r2),
        'CT':(0,1),
        'TC':(0,-1),
        'GC':(1,0),
        'CG':(-1,0)
        }

    colors = {\
        'AT':'r',
        'TA':'k',
        'GA':'g',
        'AG':'r',
        'CA':'b',
        'AC':'r',
        'GT':'g',
        'TG':'k',
        'CT':'b',
        'TC':'k',
        'GC':'g',
        'CG':'b'
        }

    label_positions = {\
        'AT':'center',
        'TA':'center',
        'GA':'center',
        'AG':'center',
        'CA':'left',
        'AC':'left',
        'GT':'left',
        'TG':'left',
        'CT':'center',
        'TC':'center',
        'GC':'center',
        'CG':'center'
        }


    def do_fontsize(k):
        return float(clip(max_text_size*sqrt(data[k]),\
            min_text_size,max_text_size))

    A = text(0,1, '$A_3$', color='r', size=do_fontsize('A'), **text_params)
    T = text(1,1, '$T_3$', color='k', size=do_fontsize('T'), **text_params)
    G = text(0,0, '$G_3$', color='g', size=do_fontsize('G'), **text_params)
    C = text(1,0, '$C_3$', color='b', size=do_fontsize('C'), **text_params)

    arrow_h_offset = 0.25  #data coordinates, empirically determined
    max_arrow_length = 1 - 2*arrow_h_offset

    max_arrow_width = max_arrow_width
    max_head_width = 2.5*max_arrow_width
    max_head_length = 2*max_arrow_width
    arrow_params={'length_includes_head':True, 'shape':shape, \
        'head_starts_at_zero':head_starts_at_zero}
    ax = gca()
    sf = 0.6   #max arrow size represents this in data coords

    d = (r2/2 + arrow_h_offset - 0.5)/r2    #distance for diags
    r2v = arrow_sep/r2                      #offset for diags

    #tuple of x, y for start position
    positions = {\
        'AT': (arrow_h_offset, 1+arrow_sep),
        'TA': (1-arrow_h_offset, 1-arrow_sep),
        'GA': (-arrow_sep, arrow_h_offset),
        'AG': (arrow_sep, 1-arrow_h_offset),
        'CA': (1-d-r2v, d-r2v),
        'AC': (d+r2v, 1-d+r2v),
        'GT': (d-r2v, d+r2v),
        'TG': (1-d+r2v, 1-d-r2v),
        'CT': (1-arrow_sep, arrow_h_offset),
        'TC': (1+arrow_sep, 1-arrow_h_offset),
        'GC': (arrow_h_offset, arrow_sep),
        'CG': (1-arrow_h_offset, -arrow_sep),
        }

    if normalize_data:
        #find maximum value for rates, i.e. where keys are 2 chars long
        max_val = 0
        for k, v in data.items():
            if len(k) == 2:
                max_val = max(max_val, v)
        #divide rates by max val, multiply by arrow scale factor
        for k, v in data.items():
            data[k] = v/max_val*sf

    def draw_arrow(pair, alpha=alpha, ec=ec, labelcolor=labelcolor):
        #set the length of the arrow
        if display == 'length':
            length = max_head_length+(max_arrow_length-max_head_length)*\
                data[pair]/sf
        else:
            length = max_arrow_length
        #set the transparency of the arrow
        if display == 'alph':
            alpha = min(data[pair]/sf, alpha)
        else:
            alpha=alpha
        #set the width of the arrow
        if display == 'width':
            scale = data[pair]/sf
            width = max_arrow_width*scale
            head_width = max_head_width*scale
            head_length = max_head_length*scale
        else:
            width = max_arrow_width
            head_width = max_head_width
            head_length = max_head_length

        fc = colors[pair]
        ec = ec or fc

        x_scale, y_scale = deltas[pair]
        x_pos, y_pos = positions[pair]
        arrow(x_pos, y_pos, x_scale*length, y_scale*length, \
            fc=fc, ec=ec, alpha=alpha, width=width, head_width=head_width, \
            head_length=head_length, **arrow_params)

        #figure out coordinates for text
        #if drawing relative to base: x and y are same as for arrow
        #dx and dy are one arrow width left and up
        #need to rotate based on direction of arrow, use x_scale and y_scale
        #as sin x and cos x?
        sx, cx = y_scale, x_scale

        where = label_positions[pair]
        if where == 'left':
            orig_position = 3*array([[max_arrow_width, max_arrow_width]])
        elif where == 'absolute':
            orig_position = array([[max_arrow_length/2.0, 3*max_arrow_width]])
        elif where == 'right':
            orig_position = array([[length-3*max_arrow_width,\
                3*max_arrow_width]])
        elif where == 'center':
            orig_position = array([[length/2.0, 3*max_arrow_width]])
        else:
            raise ValueError("Got unknown position parameter %s" % where)



        M = array([[cx, sx],[-sx,cx]])
        coords = dot(orig_position, M) + [[x_pos, y_pos]]
        x, y = ravel(coords)
        orig_label = rate_labels[pair]
        label = '$%s_{_{\mathrm{%s}}}$' % (orig_label[0], orig_label[1:])

        text(x, y, label, size=label_text_size, ha='center', va='center', \
            color=labelcolor or fc)

    for p in positions.keys():
        draw_arrow(p)

    #test data
all_on_max = dict([(i, 1) for i in 'TCAG'] + \
        [(i+j, 0.6) for i in 'TCAG' for j in 'TCAG'])

realistic_data = {
        'A':0.4,
        'T':0.3,
        'G':0.5,
        'C':0.2,
        'AT':0.4,
        'AC':0.3,
        'AG':0.2,
        'TA':0.2,
        'TC':0.3,
        'TG':0.4,
        'CT':0.2,
        'CG':0.3,
        'CA':0.2,
        'GA':0.1,
        'GT':0.4,
        'GC':0.1,
    }

extreme_data = {
        'A':0.75,
        'T':0.10,
        'G':0.10,
        'C':0.05,
        'AT':0.6,
        'AC':0.3,
        'AG':0.1,
        'TA':0.02,
        'TC':0.3,
        'TG':0.01,
        'CT':0.2,
        'CG':0.5,
        'CA':0.2,
        'GA':0.1,
        'GT':0.4,
        'GC':0.2,
    }

sample_data = {
        'A':0.2137,
        'T':0.3541,
        'G':0.1946,
        'C':0.2376,
        'AT':0.0228,
        'AC':0.0684,
        'AG':0.2056,
        'TA':0.0315,
        'TC':0.0629,
        'TG':0.0315,
        'CT':0.1355,
        'CG':0.0401,
        'CA':0.0703,
        'GA':0.1824,
        'GT':0.0387,
        'GC':0.1106,
    }


if __name__ == '__main__':
    from sys import argv
    d = None
    if len(argv) > 1:
        if argv[1] == 'full':
            d = all_on_max
            scaled = False
        elif argv[1] == 'extreme':
            d = extreme_data
            scaled = False
        elif argv[1] == 'realistic':
            d = realistic_data
            scaled = False
        elif argv[1] == 'sample':
            d = sample_data
            scaled = True
    if d is None:
        d = all_on_max
        scaled=False
    if len(argv) > 2:
        display = argv[2]
    else:
        display = 'length'

    size = 4
    figure(figsize=(size,size))

    make_arrow_plot(d, display=display, linewidth=0.001, edgecolor=None,
        normalize_data=scaled, head_starts_at_zero=True, size=size)

    draw()

    show()




********************************************
./pylab_examples/agg_buffer_to_array.py
import matplotlib.pyplot as plt
import numpy as np

# make an agg figure
fig, ax = plt.subplots()
ax.plot([1,2,3])
ax.set_title('a simple figure')
fig.canvas.draw()

# grab the pixel buffer and dump it into a numpy array
buf = fig.canvas.buffer_rgba()
l, b, w, h = fig.bbox.bounds
# The array needs to be copied, because the underlying buffer
# may be reallocated when the window is resized.
X = np.frombuffer(buf, np.uint8).copy()
X.shape = h,w,4

# now display the array X as an Axes in a new figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, frameon=False)
ax2.imshow(X)
plt.show()




********************************************
./pylab_examples/hexbin_demo2.py
"""
hexbin is an axes method or pyplot function that is essentially a
pcolor of a 2-D histogram with hexagonal cells.
"""

import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.mlab as mlab

delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

if 1:
    # make some points 20 times more common than others, but same mean
    xcond = (-1 < x) & (x < 1)
    ycond = (-2 < y) & (y < 0)
    cond = xcond & ycond
    xnew = x[cond]
    ynew = y[cond]
    znew = z[cond]
    for i in range(20):
        x = np.hstack((x,xnew))
        y = np.hstack((y,ynew))
        z = np.hstack((z,znew))

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

gridsize=30

plt.subplot(211)
plt.hexbin(x,y, C=z, gridsize=gridsize, marginals=True, cmap=plt.cm.RdBu, 
           vmax=abs(z).max(), vmin=-abs(z).max())
plt.axis([xmin, xmax, ymin, ymax])
cb = plt.colorbar()
cb.set_label('mean value')


plt.subplot(212)
plt.hexbin(x,y, gridsize=gridsize, cmap=plt.cm.Blues_r)
plt.axis([xmin, xmax, ymin, ymax])
cb = plt.colorbar()
cb.set_label('N observations')

plt.show()





********************************************
./pylab_examples/pcolor_small.py
import matplotlib.pyplot as plt
from numpy.random import rand

Z = rand(6,10)

plt.subplot(2,1,1)
c = plt.pcolor(Z)
plt.title('default: no edges')

plt.subplot(2,1,2)
c = plt.pcolor(Z, edgecolors='k', linewidths=4)
plt.title('thick edges')

plt.show()




********************************************
./pylab_examples/contourf_demo.py
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

origin = 'lower'
#origin = 'upper'

delta = 0.025

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z1 - Z2)

nr, nc = Z.shape

# put NaNs in one corner:
Z[-nr//6:, -nc//6:] = np.nan
# contourf will convert these to masked


Z = np.ma.array(Z)
# mask another corner:
Z[:nr//6, :nc//6] = np.ma.masked

# mask a circle in the middle:
interior = np.sqrt((X**2) + (Y**2)) < 0.5
Z[interior] = np.ma.masked


# We are using automatic selection of contour levels;
# this is usually not such a good idea, because they don't
# occur on nice boundaries, but we do it here for purposes
# of illustration.
CS = plt.contourf(X, Y, Z, 10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=plt.cm.bone,
                        origin=origin)

# Note that in the following, we explicitly pass in a subset of
# the contour levels used for the filled contours.  Alternatively,
# We could pass in additional levels to provide extra resolution,
# or leave out the levels kwarg to use all of the original levels.

CS2 = plt.contour(CS, levels=CS.levels[::2],
                        colors = 'r',
                        origin=origin,
                        hold='on')

plt.title('Nonsense (3 masked regions)')
plt.xlabel('word length anomaly')
plt.ylabel('sentence length anomaly')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS2)

plt.figure()

# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.
levels = [-1.5, -1, -0.5, 0, 0.5, 1]
CS3 = plt.contourf(X, Y, Z, levels,
                        colors = ('r', 'g', 'b'),
                        origin=origin,
                        extend='both')
# Our data range extends outside the range of levels; make
# data below the lowest contour level yellow, and above the
# highest level cyan:
CS3.cmap.set_under('yellow')
CS3.cmap.set_over('cyan')

CS4 = plt.contour(X, Y, Z, levels,
                       colors = ('k',),
                       linewidths = (3,),
                       origin = origin)
plt.title('Listed colors (3 masked regions)')
plt.clabel(CS4, fmt = '%2.1f', colors = 'w', fontsize=14)

# Notice that the colorbar command gets all the information it
# needs from the ContourSet object, CS3.
plt.colorbar(CS3)

# Illustrate all 4 possible "extend" settings:
extends = ["neither", "both", "min", "max"]
cmap = plt.cm.get_cmap("winter")
cmap.set_under("magenta")
cmap.set_over("yellow")
# Note: contouring simply excludes masked or nan regions, so
# instead of using the "bad" colormap value for them, it draws
# nothing at all in them.  Therefore the following would have
# no effect:
#cmap.set_bad("red")

fig, axs = plt.subplots(2,2)
for ax, extend in zip(axs.ravel(), extends):
    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
    fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.set_title("extend = %s" % extend)
    ax.locator_params(nbins=4)

plt.show()





********************************************
./pylab_examples/boxplot_demo3.py
import matplotlib.pyplot as plt
import numpy as np

def fakeBootStrapper(n):
    '''
    This is just a placeholder for the user's method of
    bootstrapping the median and its confidence intervals.

    Returns an arbitrary median and confidence intervals
    packed into a tuple
    '''
    if n == 1:
        med = 0.1
        CI = (-0.25, 0.25)
    else:
        med = 0.2
        CI = (-0.35, 0.50)

    return med, CI



np.random.seed(2)
inc = 0.1
e1 = np.random.normal(0, 1, size=(500,))
e2 = np.random.normal(0, 1, size=(500,))
e3 = np.random.normal(0, 1 + inc, size=(500,))
e4 = np.random.normal(0, 1 + 2*inc, size=(500,))

treatments = [e1,e2,e3,e4]
med1, CI1 = fakeBootStrapper(1)
med2, CI2 = fakeBootStrapper(2)
medians = [None, None, med1, med2]
conf_intervals = [None, None, CI1, CI2]

fig, ax = plt.subplots()
pos = np.array(range(len(treatments)))+1
bp = ax.boxplot(treatments, sym='k+', positions=pos,
                notch=1, bootstrap=5000,
                usermedians=medians,
                conf_intervals=conf_intervals)

ax.set_xlabel('treatment')
ax.set_ylabel('response')
plt.setp(bp['whiskers'], color='k',  linestyle='-' )
plt.setp(bp['fliers'], markersize=3.0)
plt.show()




********************************************
./pylab_examples/demo_tight_layout.py

import matplotlib.pyplot as plt


import random
fontsizes = [8, 16, 24, 32]
def example_plot(ax):
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=random.choice(fontsizes))
    ax.set_ylabel('y-label', fontsize=random.choice(fontsizes))
    ax.set_title('Title', fontsize=random.choice(fontsizes))

fig, ax = plt.subplots()
example_plot(ax)
plt.tight_layout()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
example_plot(ax1)
example_plot(ax2)
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
example_plot(ax1)
example_plot(ax2)
plt.tight_layout()

fig, axes = plt.subplots(nrows=3, ncols=3)
for row in axes:
    for ax in row:
        example_plot(ax)
plt.tight_layout()


fig = plt.figure()

ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)

plt.tight_layout()


fig = plt.figure()

ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)

plt.tight_layout()

plt.show()


fig = plt.figure()

import matplotlib.gridspec as gridspec

gs1 = gridspec.GridSpec(3, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[2])

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)

gs1.tight_layout(fig, rect=[None, None, 0.45, None])

gs2 = gridspec.GridSpec(2, 1)
ax4 = fig.add_subplot(gs2[0])
ax5 = fig.add_subplot(gs2[1])

#example_plot(ax4)
#example_plot(ax5)

gs2.tight_layout(fig, rect=[0.45, None, None, None])

# now match the top and bottom of two gridspecs.
top = min(gs1.top, gs2.top)
bottom = max(gs1.bottom, gs2.bottom)

gs1.update(top=top, bottom=bottom)
gs2.update(top=top, bottom=bottom)

plt.show()







********************************************
./pylab_examples/subplot_demo.py
"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'ko-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()




********************************************
./pylab_examples/manual_axis.py
"""
The techniques here are no longer required with the new support for
spines in matplotlib -- see
http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html.

This example should be considered deprecated and is left just for demo
purposes for folks wanting to make a pseudo-axis

"""

import numpy as np
from pylab import figure, show
import matplotlib.lines as lines

def make_xaxis(ax, yloc, offset=0.05, **props):
    xmin, xmax = ax.get_xlim()
    locs = [loc for loc in ax.xaxis.get_majorticklocs()
            if loc>=xmin and loc<=xmax]
    tickline, = ax.plot(locs, [yloc]*len(locs),linestyle='',
            marker=lines.TICKDOWN, **props)
    axline, = ax.plot([xmin, xmax], [yloc, yloc], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)
    for loc in locs:
        ax.text(loc, yloc-offset, '%1.1f'%loc,
                horizontalalignment='center',
                verticalalignment='top')

def make_yaxis(ax, xloc=0, offset=0.05, **props):
    ymin, ymax = ax.get_ylim()
    locs = [loc for loc in ax.yaxis.get_majorticklocs()
            if loc>=ymin and loc<=ymax]
    tickline, = ax.plot([xloc]*len(locs), locs, linestyle='',
            marker=lines.TICKLEFT, **props)
    axline, = ax.plot([xloc, xloc], [ymin, ymax], **props)
    tickline.set_clip_on(False)
    axline.set_clip_on(False)

    for loc in locs:
        ax.text(xloc-offset, loc, '%1.1f'%loc,
                verticalalignment='center',
                horizontalalignment='right')


props = dict(color='black', linewidth=2, markeredgewidth=2)
x = np.arange(200.)
y = np.sin(2*np.pi*x/200.) + np.random.rand(200)-0.5
fig = figure(facecolor='white')
ax = fig.add_subplot(111, frame_on=False)
ax.axison = False
ax.plot(x, y, 'd', markersize=8, markerfacecolor='blue')
ax.set_xlim(0, 200)
ax.set_ylim(-1.5, 1.5)
make_xaxis(ax, 0, offset=0.1, **props)
make_yaxis(ax, 0, offset=5, **props)

show()




********************************************
./pylab_examples/line_collection2.py
from pylab import *
from matplotlib.collections import LineCollection

# In order to efficiently plot many lines in a single set of axes,
# Matplotlib has the ability to add the lines all at once. Here is a
# simple example showing how it is done.

N = 50
x = arange(N)
# Here are many sets of y to plot vs x
ys = [x+i for i in x]

# We need to set the plot limits, they will not autoscale
ax = axes()
ax.set_xlim((amin(x),amax(x)))
ax.set_ylim((amin(amin(ys)),amax(amax(ys))))

# colors is sequence of rgba tuples
# linestyle is a string or dash tuple. Legal string values are
#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
#          where onoffseq is an even length tuple of on and off ink in points.
#          If linestyle is omitted, 'solid' is used
# See matplotlib.collections.LineCollection for more information
line_segments = LineCollection([list(zip(x,y)) for y in ys], # Make a sequence of x,y pairs
                                linewidths    = (0.5,1,1.5,2),
                                linestyles = 'solid')
line_segments.set_array(x)
ax.add_collection(line_segments)
fig = gcf()
axcb = fig.colorbar(line_segments)
axcb.set_label('Line Number')
ax.set_title('Line Collection with mapped colors')
sci(line_segments) # This allows interactive changing of the colormap.
show()




********************************************
./pylab_examples/eventcollection_demo.py
#!/usr/bin/env python
# -*- Coding:utf-8 -*-
'''Plot two curves, then use EventCollections to mark the locations of the x
and y data points on the respective axes for each curve'''

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

# create random data
np.random.seed(50)
xdata = np.random.random([2, 10])

# split the data into two parts
xdata1 = xdata[0, :]
xdata2 = xdata[1, :]

# sort the data so it makes clean curves
xdata1.sort()
xdata2.sort()

# create some y data points
ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 3

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata1, ydata1, 'r', xdata2, ydata2, 'b')

# create the events marking the x data points
xevents1 = EventCollection(xdata1, color=[1, 0, 0], linelength=0.05)
xevents2 = EventCollection(xdata2, color=[0, 0, 1], linelength=0.05)

# create the events marking the y data points
yevents1 = EventCollection(ydata1, color=[1, 0, 0], linelength=0.05,
                           orientation='vertical')
yevents2 = EventCollection(ydata2, color=[0, 0, 1], linelength=0.05,
                           orientation='vertical')

# add the events to the axis
ax.add_collection(xevents1)
ax.add_collection(xevents2)
ax.add_collection(yevents1)
ax.add_collection(yevents2)

# set the limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax.set_title('line plot with data points')

# display the plot
plt.show()




********************************************
./pylab_examples/ginput_demo.py
# -*- noplot -*-

from __future__ import print_function

from pylab import arange, plot, sin, ginput, show
t = arange(10)
plot(t, sin(t))
print("Please click")
x = ginput(3)
print("clicked",x)
show()




********************************************
./pylab_examples/gradient_bar.py
from matplotlib.pyplot import figure, show, cm
from numpy import arange
from numpy.random import rand

def gbar(ax, x, y, width=0.5, bottom=0):
   X = [[.6, .6],[.7,.7]]
   for left,top in zip(x, y):
       right = left+width
       ax.imshow(X, interpolation='bicubic', cmap=cm.Blues,
                 extent=(left, right, bottom, top), alpha=1)

fig = figure()

xmin, xmax = xlim = 0,10
ymin, ymax = ylim = 0,1
ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                    autoscale_on=False)
X = [[.6, .6],[.7,.7]]

ax.imshow(X, interpolation='bicubic', cmap=cm.copper,
         extent=(xmin, xmax, ymin, ymax), alpha=1)

N = 10
x = arange(N)+0.25
y = rand(N)
gbar(ax, x, y, width=0.7)
ax.set_aspect('auto')
show()




********************************************
./pylab_examples/barchart_demo.py
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt


n_groups = 5

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=std_men,
                 error_kw=error_config,
                 label='Men')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_women,
                 error_kw=error_config,
                 label='Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))
plt.legend()

plt.tight_layout()
plt.show()




********************************************
./pylab_examples/mri_demo.py
#!/usr/bin/env python

from __future__ import print_function
from pylab import *
import matplotlib.cbook as cbook
# data are 256x256 16 bit integers
dfile = cbook.get_sample_data('s1045.ima.gz')
im = np.fromstring(dfile.read(), np.uint16).astype(float)
im.shape = 256, 256

#imshow(im, ColormapJet(256))
imshow(im, cmap=cm.gray)
axis('off')

show()




********************************************
./pylab_examples/tex_demo.py
"""
You can use TeX to render all of your matplotlib text if the rc
parameter text.usetex is set.  This works currently on the agg and ps
backends, and requires that you have tex and the other dependencies
described at http://matplotlib.org/users/usetex.html
properly installed on your system.  The first time you run a script
you will see a lot of output from tex and associated tools.  The next
time, the run may be silent, as a lot of the information is cached in
~/.tex.cache

"""
import numpy as np
import matplotlib.pyplot as plt


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(1, figsize=(6,4))
ax = plt.axes([0.1, 0.1, 0.8, 0.7])
t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2
plt.plot(t, s)

plt.xlabel(r'\textbf{time (s)}')
plt.ylabel(r'\textit{voltage (mV)}',fontsize=16)
plt.title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty"
          r"\frac{-e^{i\pi}}{2^n}$!", fontsize=16, color='r')
plt.grid(True)
plt.savefig('tex_demo')
plt.show()




********************************************
./pylab_examples/trigradient_demo.py
"""
Demonstrates computation of gradient with matplotlib.tri.CubicTriInterpolator.
"""
from matplotlib.tri import Triangulation, UniformTriRefiner,\
    CubicTriInterpolator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math


#-----------------------------------------------------------------------------
# Electrical potential of a dipole
#-----------------------------------------------------------------------------
def dipole_potential(x, y):
    """ The electric dipole potential V """
    r_sq = x**2 + y**2
    theta = np.arctan2(y, x)
    z = np.cos(theta)/r_sq
    return (np.max(z)-z) / (np.max(z)-np.min(z))


#-----------------------------------------------------------------------------
# Creating a Triangulation
#-----------------------------------------------------------------------------
# First create the x and y coordinates of the points.
n_angles = 30
n_radii = 10
min_radius = 0.2
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
V = dipole_potential(x, y)

# Create the Triangulation; no triangles specified so Delaunay triangulation
# created.
triang = Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

#-----------------------------------------------------------------------------
# Refine data - interpolates the electrical potential V
#-----------------------------------------------------------------------------
refiner = UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(V, subdiv=3)

#-----------------------------------------------------------------------------
# Computes the electrical field (Ex, Ey) as gradient of electrical potential
#-----------------------------------------------------------------------------
tci = CubicTriInterpolator(triang, -V)
# Gradient requested here at the mesh nodes but could be anywhere else:
(Ex, Ey) = tci.gradient(triang.x, triang.y)
E_norm = np.sqrt(Ex**2 + Ey**2)

#-----------------------------------------------------------------------------
# Plot the triangulation, the potential iso-contours and the vector field
#-----------------------------------------------------------------------------
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(triang, color='0.8')

levels = np.arange(0., 1., 0.01)
cmap = cm.get_cmap(name='hot', lut=None)
plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
               linewidths=[2.0, 1.0, 1.0, 1.0])
# Plots direction of the electrical vector field
plt.quiver(triang.x, triang.y, Ex/E_norm, Ey/E_norm,
           units='xy', scale=10., zorder=3, color='blue',
           width=0.007, headwidth=3., headlength=4.)

plt.title('Gradient plot: an electrical dipole')
plt.show()




********************************************
./pylab_examples/filledmarker_demo.py
import itertools

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
altcolor = 'lightgreen'

plt.rcParams['text.usetex'] = False # otherwise, '^' will cause trouble

y = np.arange(10)
for marker in mlines.Line2D.filled_markers:
    f = plt.figure()
    f.text(.5,.95, "marker = %r" % marker, ha='center')
    for i,fs in enumerate(mlines.Line2D.fillStyles):
        color = colors.next()

        ax = f.add_subplot(121)
        ax.plot(2*(4-i)+y, c=color,
                       marker=marker,
                       markersize=20, 
                       fillstyle=fs, 
                       label=fs)
        ax.legend(loc=2)
        ax.set_title('fillstyle')

        ax = f.add_subplot(122)
        ax.plot(2*(4-i)+y, c=color,
                       marker=marker,
                       markersize=20,
                       markerfacecoloralt=altcolor,
                       fillstyle=fs,
                       label=fs)
        ax.legend(loc=2)
        ax.set_title('fillstyle')

plt.show()




********************************************
./pylab_examples/errorbar_limits.py
'''
Illustration of upper and lower limit symbols on errorbars
'''

from math import pi
from numpy import array, arange, sin
import pylab as P

fig = P.figure()
x = arange(10.0)
y = sin(arange(10.0)/20.0*pi)

P.errorbar(x,y,yerr=0.1,capsize=3)

y = sin(arange(10.0)/20.0*pi) + 1
P.errorbar(x,y,yerr=0.1, uplims=True)

y = sin(arange(10.0)/20.0*pi) + 2
upperlimits = array([1,0]*5)
lowerlimits = array([0,1]*5)
P.errorbar(x, y, yerr=0.1, uplims=upperlimits, lolims=lowerlimits)

P.xlim(-1,10)

fig = P.figure()
x = arange(10.0)/10.0
y = (x+0.1)**2

P.errorbar(x, y, xerr=0.1, xlolims=True)
y = (x+0.1)**3

P.errorbar(x+0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits)

y = (x+0.1)**4
P.errorbar(x+1.2, y, xerr=0.1, xuplims=True)

P.xlim(-0.2,2.4)
P.ylim(-0.1,1.3)

P.show()




********************************************
./pylab_examples/stock_demo.py
#!/usr/bin/env python

from matplotlib.ticker import MultipleLocator
from pylab import *
from data_helper import get_two_stock_data

d1, p1, d2, p2 = get_two_stock_data()

fig, ax = subplots()
lines = plot(d1, p1, 'bs', d2, p2, 'go')
xlabel('Days')
ylabel('Normalized price')
xlim(0, 3)
ax.xaxis.set_major_locator(MultipleLocator(1))

title('INTC vs AAPL')
legend( ('INTC', 'AAPL') )

show()




********************************************
./pylab_examples/toggle_images.py
#!/usr/bin/env python
""" toggle between two images by pressing "t"

The basic idea is to load two images (they can be different shapes) and plot
them to the same axes with hold "on".  Then, toggle the visible property of
them using keypress event handling

If you want two images with different shapes to be plotted with the same
extent, they must have the same "extent" property

As usual, we'll define some random images for demo.  Real data is much more
exciting!

Note, on the wx backend on some platforms (eg linux), you have to
first click on the figure before the keypress events are activated.
If you know how to fix this, please email us!

"""

from pylab import *

# two images x1 is initially visible, x2 is not
x1 = rand(100, 100)
x2 = rand(150, 175)

# arbitrary extent - both images must have same extent if you want
# them to be resampled into the same axes space
extent = (0,1,0,1)
im1 = imshow(x1, extent=extent)
im2 = imshow(x2, extent=extent, hold=True)
im2.set_visible(False)

def toggle_images(event):
    'toggle the visible state of the two images'
    if event.key != 't': return
    b1 = im1.get_visible()
    b2 = im2.get_visible()
    im1.set_visible(not b1)
    im2.set_visible(not b2)
    draw()

connect('key_press_event', toggle_images)

show()




********************************************
./pylab_examples/hexbin_demo.py
"""
hexbin is an axes method or pyplot function that is essentially
a pcolor of a 2-D histogram with hexagonal cells.  It can be
much more informative than a scatter plot; in the first subplot
below, try substituting 'scatter' for 'hexbin'.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

plt.subplots_adjust(hspace=0.5)
plt.subplot(121)
plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)
plt.axis([xmin, xmax, ymin, ymax])
plt.title("Hexagon binning")
cb = plt.colorbar()
cb.set_label('counts')

plt.subplot(122)
plt.hexbin(x,y,bins='log', cmap=plt.cm.YlOrRd_r)
plt.axis([xmin, xmax, ymin, ymax])
plt.title("With a log color scale")
cb = plt.colorbar()
cb.set_label('log10(N)')

plt.show()




********************************************
./pylab_examples/multiple_figs_demo.py
#!/usr/bin/env python
# Working with multiple figure windows and subplots
from pylab import *

t = arange(0.0, 2.0, 0.01)
s1 = sin(2*pi*t)
s2 = sin(4*pi*t)

figure(1)
subplot(211)
plot(t,s1)
subplot(212)
plot(t,2*s1)

figure(2)
plot(t,s2)

# now switch back to figure 1 and make some changes
figure(1)
subplot(211)
plot(t,s2, 'gs')
setp(gca(), 'xticklabels', [])

figure(1)
savefig('fig1')
figure(2)
savefig('fig2')

show()




********************************************
./pylab_examples/centered_ticklabels.py
# sometimes it is nice to have ticklabels centered.  mpl currently
# associates a label with a tick, and the label can be aligned
# 'center', 'left', or 'right' using the horizontal alignment property:
#
#
#   for label in ax.xaxis.get_xticklabels():
#       label.set_horizontalalignment('right')
#
#
# but this doesn't help center the label between ticks.  One solution
# is to "face it".  Use the minor ticks to place a tick centered
# between the major ticks.  Here is an example that labels the months,
# centered between the ticks

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# load some financial data; apple's stock price
fh = cbook.get_sample_data('aapl.npy.gz')
r = np.load(fh); fh.close()
r = r[-250:]  # get the last 250 days

fig, ax = plt.subplots()
ax.plot(r.date, r.adj_close)

ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))

for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

imid = len(r)/2
ax.set_xlabel(str(r.date[imid].year))
plt.show()




********************************************
./pylab_examples/histogram_demo_extended.py
#!/usr/bin/env python
import numpy as np
import pylab as P

#
# The hist() function now has a lot more options
#

#
# first create a single histogram
#
mu, sigma = 200, 25
x = mu + sigma*P.randn(10000)

# the histogram of the data with histtype='step'
n, bins, patches = P.hist(x, 50, normed=1, histtype='stepfilled')
P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a line showing the expected distribution
y = P.normpdf( bins, mu, sigma)
l = P.plot(bins, y, 'k--', linewidth=1.5)


#
# create a histogram by providing the bin edges (unequally spaced)
#
P.figure()

bins = [100,125,150,160,170,180,190,200,210,220,230,240,250,275,300]
# the histogram of the data with histtype='step'
n, bins, patches = P.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)

#
# now we create a cumulative histogram of the data
#
P.figure()

n, bins, patches = P.hist(x, 50, normed=1, histtype='step', cumulative=True)

# add a line showing the expected distribution
y = P.normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
l = P.plot(bins, y, 'k--', linewidth=1.5)

# create a second data-set with a smaller standard deviation
sigma2 = 15.
x = mu + sigma2*P.randn(10000)

n, bins, patches = P.hist(x, bins=bins, normed=1, histtype='step', cumulative=True)

# add a line showing the expected distribution
y = P.normpdf( bins, mu, sigma2).cumsum()
y /= y[-1]
l = P.plot(bins, y, 'r--', linewidth=1.5)

# finally overplot a reverted cumulative histogram
n, bins, patches = P.hist(x, bins=bins, normed=1,
    histtype='step', cumulative=-1)


P.grid(True)
P.ylim(0, 1.05)


#
# histogram has the ability to plot multiple data in parallel ...
# Note the new color kwarg, used to override the default, which
# uses the line color cycle.
#
P.figure()

# create a new data-set
x = mu + sigma*P.randn(1000,3)

n, bins, patches = P.hist(x, 10, normed=1, histtype='bar',
                            color=['crimson', 'burlywood', 'chartreuse'],
                            label=['Crimson', 'Burlywood', 'Chartreuse'])
P.legend()

#
# ... or we can stack the data
#
P.figure()

n, bins, patches = P.hist(x, 10, normed=1, histtype='bar', stacked=True)

P.show()

#
# we can also stack using the step histtype
#

P.figure()

n, bins, patches = P.hist(x, 10, histtype='step', stacked=True, fill=True)

P.show()

#
# finally: make a multiple-histogram of data-sets with different length
#
x0 = mu + sigma*P.randn(10000)
x1 = mu + sigma*P.randn(7000)
x2 = mu + sigma*P.randn(3000)

# and exercise the weights option by arbitrarily giving the first half
# of each series only half the weight of the others:

w0 = np.ones_like(x0)
w0[:len(x0)/2] = 0.5
w1 = np.ones_like(x1)
w1[:len(x1)/2] = 0.5
w2 = np.ones_like(x2)
w2[:len(x2)/2] = 0.5



P.figure()

n, bins, patches = P.hist( [x0,x1,x2], 10, weights=[w0, w1, w2], histtype='bar')

P.show()




********************************************
./pylab_examples/multipage_pdf.py
# This is a demo of creating a pdf file with several pages.

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x*x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = u'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()




********************************************
./pylab_examples/matshow.py
"""Simple matshow() example."""
from matplotlib.pylab import *


def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa


# Display 2 matrices of different sizes
dimlist = [(12, 12), (15, 35)]
for d in dimlist:
    matshow(samplemat(d))

# Display a random matrix with a specified figure number and a grayscale
# colormap
matshow(rand(64, 64), fignum=100, cmap=cm.gray)

show()




********************************************
./pylab_examples/invert_axes.py
#!/usr/bin/env python
"""

You can use decreasing axes by flipping the normal order of the axis
limits

"""
from pylab import *

t = arange(0.01, 5.0, 0.01)
s = exp(-t)
plot(t, s)

xlim(5,0)  # decreasing time

xlabel('decreasing time (s)')
ylabel('voltage (mV)')
title('Should be growing...')
grid(True)

show()




********************************************
./pylab_examples/logo.py
#!/usr/bin/env python
# This file generates the matplotlib web page logo

from __future__ import print_function
from pylab import *
import matplotlib.cbook as cbook

# convert data to mV
datafile = cbook.get_sample_data('membrane.dat', asfileobj=False)
print('loading', datafile)

x = 1000*0.1*fromstring(file(datafile, 'rb').read(), float32)
# 0.0005 is the sample interval
t = 0.0005*arange(len(x))
figure(1, figsize=(7,1), dpi=100)
ax = subplot(111, axisbg='y')
plot(t, x)
text(0.5, 0.5,'matplotlib', color='r',
     fontsize=40, fontname='Courier',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes,
     )
axis([1, 1.72,-60, 10])
setp(gca(), 'xticklabels', [])
setp(gca(), 'yticklabels', [])

show()




********************************************
./pylab_examples/stackplot_demo2.py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
def layers(n, m):
    """
    Return *n* random Gaussian mixtures, each of length *m*.
    """
    def bump(a):
        x = 1 / (.1 + np.random.random())
        y = 2 * np.random.random() - .5
        z = 10 / (.1 + np.random.random())
        for i in range(m):
            w = (i / float(m) - y) * z
            a[i] += x * np.exp(-w * w)
    a = np.zeros((m, n))
    for i in range(n):
        for j in range(5):
            bump(a[:, i])
    return a

d = layers(3, 100)

plt.subplots()
plt.stackplot(range(100), d.T, baseline='wiggle')
plt.show()




********************************************
./pylab_examples/axhspan_demo.py
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-1,2, .01)
s = np.sin(2*np.pi*t)

plt.plot(t,s)
# draw a thick red hline at y=0 that spans the xrange
l = plt.axhline(linewidth=4, color='r')

# draw a default hline at y=1 that spans the xrange
l = plt.axhline(y=1)

# draw a default vline at x=1 that spans the yrange
l = plt.axvline(x=1)

# draw a thick blue vline at x=0 that spans the upper quadrant of
# the yrange
l = plt.axvline(x=0, ymin=0.75, linewidth=4, color='b')

# draw a default hline at y=.5 that spans the the middle half of
# the axes
l = plt.axhline(y=.5, xmin=0.25, xmax=0.75)

p = plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

p = plt.axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

plt.axis([-1,2,-1,2])


plt.show()




********************************************
./pylab_examples/image_interp.py
#!/usr/bin/env python
"""
The same (small) array, interpolated with three different
interpolation methods.

The center of the pixel at A[i,j] is plotted at i+0.5, i+0.5.  If you
are using interpolation='nearest', the region bounded by (i,j) and
(i+1,j+1) will have the same color.  If you are using interpolation,
the pixel center will have the same color as it does with nearest, but
other pixels will be interpolated between the neighboring pixels.

Earlier versions of matplotlib (<0.63) tried to hide the edge effects
from you by setting the view limits so that they would not be visible.
A recent bugfix in antigrain, and a new implementation in the
matplotlib._image module which takes advantage of this fix, no longer
makes this necessary.  To prevent edge effects, when doing
interpolation, the matplotlib._image module now pads the input array
with identical pixels around the edge.  e.g., if you have a 5x5 array
with colors a-y as below


  a b c d e
  f g h i j
  k l m n o
  p q r s t
  u v w x y

the _image module creates the padded array,

  a a b c d e e
  a a b c d e e
  f f g h i j j
  k k l m n o o
  p p q r s t t
  o u v w x y y
  o u v w x y y

does the interpolation/resizing, and then extracts the central region.
This allows you to plot the full range of your array w/o edge effects,
and for example to layer multiple images of different sizes over one
another with different interpolation methods - see
examples/layer_images.py.  It also implies a performance hit, as this
new temporary, padded array must be created.  Sophisticated
interpolation also implies a performance hit, so if you need maximal
performance or have very large images, interpolation='nearest' is
suggested.

"""
from pylab import *
A = rand(5,5)
figure(1)
imshow(A, interpolation='nearest')
grid(True)

figure(2)
imshow(A, interpolation='bilinear')
grid(True)

figure(3)
imshow(A, interpolation='bicubic')
grid(True)

show()




********************************************
./pylab_examples/usetex_fonteffects.py
# This script demonstrates that font effects specified in your pdftex.map
# are now supported in pdf usetex.

import matplotlib
matplotlib.rc('text', usetex=True)
import pylab

def setfont(font):
    return r'\font\a %s at 14pt\a ' % font

for y, font, text in zip(range(5),
                         ['ptmr8r', 'ptmri8r', 'ptmro8r', 'ptmr8rn', 'ptmrr8re'],
                         ['Nimbus Roman No9 L ' + x for x in
                          ['', 'Italics (real italics for comparison)',
                           '(slanted)', '(condensed)', '(extended)']]):
    pylab.text(0, y, setfont(font) + text)

pylab.ylim(-1, 5)
pylab.xlim(-0.2, 0.6)
pylab.setp(pylab.gca(), frame_on=False, xticks=(), yticks=())
pylab.title('Usetex font effects')
pylab.savefig('usetex_fonteffects.pdf')




********************************************
./pylab_examples/log_demo.py

import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(hspace=0.4)
t = np.arange(0.01, 20.0, 0.01)

# log y axis
plt.subplot(221)
plt.semilogy(t, np.exp(-t/5.0))
plt.title('semilogy')
plt.grid(True)

# log x axis
plt.subplot(222)
plt.semilogx(t, np.sin(2*np.pi*t))
plt.title('semilogx')
plt.grid(True)

# log x and y axis
plt.subplot(223)
plt.loglog(t, 20*np.exp(-t/10.0), basex=2)
plt.grid(True)
plt.title('loglog base 4 on x')

# with errorbars: clip non-positive values
ax = plt.subplot(224)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')

x = 10.0**np.linspace(0.0, 2.0, 20)
y = x**2.0
plt.errorbar(x, y, xerr=0.1*x, yerr=5.0+0.75*y)
ax.set_ylim(ymin=0.1)
ax.set_title('Errorbars go negative')


plt.show()




********************************************
./pylab_examples/demo_annotation_box.py
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
     AnnotationBbox
from matplotlib.cbook import get_sample_data

import numpy as np

if 1:
    fig, ax = plt.subplots()

    offsetbox = TextArea("Test 1", minimumdescent=False)

    xy = (0.5, 0.7)

    ax.plot(xy[0], xy[1], ".r")

    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(-20, 40),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    offsetbox = TextArea("Test", minimumdescent=False)

    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(1.02, xy[1]),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0.,0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


    from matplotlib.patches import Circle
    da = DrawingArea(20, 20, 0, 0)
    p = Circle((10, 10), 10)
    da.add_artist(p)

    xy = [0.3, 0.55]
    ab = AnnotationBbox(da, xy,
                        xybox=(1.02, xy[1]),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0.,0.5),
                        arrowprops=dict(arrowstyle="->"))
                        #arrowprops=None)

    ax.add_artist(ab)


    arr = np.arange(100).reshape((10,10))
    im = OffsetImage(arr, zoom=2)

    ab = AnnotationBbox(im, xy,
                        xybox=(-50., 50.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.3,
                        arrowprops=dict(arrowstyle="->"))
                        #arrowprops=None)

    ax.add_artist(ab)


    # another image


    from matplotlib._png import read_png
    fn = get_sample_data("grace_hopper.png", asfileobj=False)
    arr_lena = read_png(fn)

    imagebox = OffsetImage(arr_lena, zoom=0.2)

    ab = AnnotationBbox(imagebox, xy,
                        xybox=(120., -80.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )


    ax.add_artist(ab)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    plt.draw()
    plt.show()




********************************************
./pylab_examples/plotfile_demo.py
from pylab import plotfile, show, gca
import matplotlib.cbook as cbook

fname = cbook.get_sample_data('msft.csv', asfileobj=False)
fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)

# test 1; use ints
plotfile(fname, (0,5,6))

# test 2; use names
plotfile(fname, ('date', 'volume', 'adj_close'))

# test 3; use semilogy for volume
plotfile(fname, ('date', 'volume', 'adj_close'), plotfuncs={'volume': 'semilogy'})

# test 4; use semilogy for volume
plotfile(fname, (0,5,6), plotfuncs={5:'semilogy'})

#test 5; single subplot
plotfile(fname, ('date', 'open', 'high', 'low', 'close'), subplots=False)

# test 6; labeling, if no names in csv-file
plotfile(fname2, cols=(0,1,2), delimiter=' ',
         names=['$x$', '$f(x)=x^2$', '$f(x)=x^3$'])

# test 7; more than one file per figure--illustrated here with a single file
plotfile(fname2, cols=(0, 1), delimiter=' ')
plotfile(fname2, cols=(0, 2), newfig=False, delimiter=' ') # use current figure
gca().set_xlabel(r'$x$')
gca().set_ylabel(r'$f(x) = x^2, x^3$')

# test 8; use bar for volume
plotfile(fname, (0,5,6), plotfuncs={5:'bar'})

show()






********************************************
./pylab_examples/contour_demo.py
#!/usr/bin/env python
"""
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)



# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')


# contour labels can be placed manually by providing list of positions
# (in data coordinate). See ginput_manual_clabel.py for interactive
# placement.
plt.figure()
CS = plt.contour(X, Y, Z)
manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
plt.title('labels at selected locations')


# You can force all the contours to be the same color.
plt.figure()
CS = plt.contour(X, Y, Z, 6,
                 colors='k', # negative contours will be dashed by default
                 )
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Single color - negative contours dashed')

# You can set negative contours to be solid instead of dashed:
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.figure()
CS = plt.contour(X, Y, Z, 6,
                 colors='k', # negative contours will be dashed by default
                 )
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Single color - negative contours solid')


# And you can manually specify the colors of the contour
plt.figure()
CS = plt.contour(X, Y, Z, 6,
                 linewidths=np.arange(.5, 4, .5),
                 colors=('r', 'green', 'blue', (1,1,0), '#afeeee', '0.5')
                 )
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Crazy lines')


# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
plt.figure()
im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(-3,3,-2,2))
levels = np.arange(-1.2, 1.6, 0.2)
CS = plt.contour(Z, levels,
                 origin='lower',
                 linewidths=2,
                 extent=(-3,3,-2,2))

#Thicken the zero contour.
zc = CS.collections[6]
plt.setp(zc, linewidth=4)

plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=14)

# make a colorbar for the contour lines
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('Lines with colorbar')
#plt.hot()  # Now change the colormap for the contour lines and colorbar
plt.flag()

# We can still add a colorbar for the image, too.
CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

l,b,w,h = plt.gca().get_position().bounds
ll,bb,ww,hh = CB.ax.get_position().bounds
CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])


plt.show()




********************************************
./pylab_examples/hist_colormapped.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

fig, ax = plt.subplots()
Ntotal = 1000
N, bins, patches = ax.hist(np.random.rand(Ntotal), 20)

#I'll color code by height, but you could use any scalar


# we need to normalize the data to 0..1 for the full
# range of the colormap
fracs = N.astype(float)/N.max()
norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = cm.jet(norm(thisfrac))
    thispatch.set_facecolor(color)


plt.show()




********************************************
./pylab_examples/anchored_artists.py
from matplotlib.patches import Rectangle, Ellipse

from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker,\
     TextArea, DrawingArea


class AnchoredText(AnchoredOffsetbox):
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):

        self.txt = TextArea(s,
                            minimumdescent=False)


        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad,
                                           child=self.txt,
                                           prop=prop,
                                           frameon=frameon)


class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0,0), size, 0, fc="none"))

        self.txt_label = TextArea(label, minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(self, transform, width, height, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=True):
        """
        Draw an ellipse the size in data coordinate of the give axes.

        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse((0,0), width, height, angle)
        self._box.add_artist(self.ellipse)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)



class AnchoredDrawingArea(AnchoredOffsetbox):
    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):

        self.da = DrawingArea(width, height, xdescent, ydescent, clip=True)

        super(AnchoredDrawingArea, self).__init__(loc, pad=pad, borderpad=borderpad,
                                                  child=self.da,
                                                  prop=None,
                                                  frameon=frameon)



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.set_aspect(1.)

    at = AnchoredText("Figure 1a",
                      loc=2, frameon=True)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    from matplotlib.patches import Circle
    ada = AnchoredDrawingArea(20, 20, 0, 0,
                              loc=1, pad=0., frameon=False)
    p = Circle((10, 10), 10)
    ada.da.add_artist(p)
    ax.add_artist(ada)

    # draw an ellipse of width=0.1, height=0.15 in the data coordinate
    ae = AnchoredEllipse(ax.transData, width=0.1, height=0.15, angle=0.,
                         loc=3, pad=0.5, borderpad=0.4, frameon=True)

    ax.add_artist(ae)

    # draw a horizontal bar with length of 0.1 in Data coordinate
    # (ax.transData) with a label underneath.
    asb =  AnchoredSizeBar(ax.transData,
                          0.1,
                          r"1$^{\prime}$",
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)

    plt.draw()
    plt.show()








********************************************
./pylab_examples/polar_legend.py
#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import figure, show, rc

# radar green, solid grid lines
rc('grid', color='#316931', linewidth=1, linestyle='-')
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)

# force square figure and square axes looks better for polar, IMO
fig = figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='#d5de9c')

r = np.arange(0, 3.0, 0.01)
theta = 2*np.pi*r
ax.plot(theta, r, color='#ee8d18', lw=3, label='a line')
ax.plot(0.5*theta, r, color='blue', ls='--', lw=3, label='another line')
ax.legend()

show()




********************************************
./pylab_examples/image_slices_viewer.py
from __future__ import print_function
import numpy
from matplotlib.pyplot import figure, show




class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows,cols,self.slices = X.shape
        self.ind  = self.slices/2

        self.im = ax.imshow(self.X[:,:,self.ind])
        self.update()

    def onscroll(self, event):
        print ("%s %s" % (event.button, event.step))
        if event.button=='up':
            self.ind = numpy.clip(self.ind+1, 0, self.slices-1)
        else:
            self.ind = numpy.clip(self.ind-1, 0, self.slices-1)


        self.update()

    def update(self):
        self.im.set_data(self.X[:,:,self.ind])
        ax.set_ylabel('slice %s'%self.ind)
        self.im.axes.figure.canvas.draw()


fig = figure()
ax = fig.add_subplot(111)

X = numpy.random.rand(20,20,40)

tracker = IndexTracker(ax, X)



fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
show()




********************************************
./pylab_examples/date_demo_rrule.py
#!/usr/bin/env python
"""
Show how to use an rrule instance to make a custom date ticker - here
we put a tick mark on every 5th easter

See https://moin.conectiva.com.br/DateUtil for help with rrules
"""
import matplotlib.pyplot as plt
from matplotlib.dates import YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange
import numpy as np
import datetime

# tick every 5th easter
rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
loc = RRuleLocator(rule)
formatter = DateFormatter('%m/%d/%y')
date1 = datetime.date( 1952, 1, 1 )
date2 = datetime.date( 2004, 4, 12 )
delta = datetime.timedelta(days=100)

dates = drange(date1, date2, delta)
s = np.random.rand(len(dates)) # make up some random y values


fig, ax = plt.subplots()
plt.plot_date(dates, s)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(formatter)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=30, fontsize=10)

plt.show()




********************************************
./pylab_examples/shared_axis_across_figures.py
"""
connect the data limits on the axes in one figure with the axes in
another.  This is not the right way to do this for two axes in the
same figure -- use the sharex and sharey property in that case
"""
# -*- noplot -*-
import numpy
from pylab import figure, show

fig1 = figure()
fig2 = figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111, sharex=ax1, sharey=ax1)

ax1.plot(numpy.random.rand(100), 'o')
ax2.plot(numpy.random.rand(100), 'o')

# In the latest release, it is no longer necessary to do anything
# special to share axes across figures:

# ax1.sharex_foreign(ax2)
# ax2.sharex_foreign(ax1)

# ax1.sharey_foreign(ax2)
# ax2.sharey_foreign(ax1)

show()




********************************************
./pylab_examples/tricontour_smooth_user.py
"""
Demonstrates high-resolution tricontouring on user-defined triangular grids
with matplotlib.tri.UniformTriRefiner
"""
from matplotlib.tri import Triangulation, UniformTriRefiner
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math


#-----------------------------------------------------------------------------
# Analytical test function
#-----------------------------------------------------------------------------
def function_z(x, y):
    """ A function of 2 variables """
    r1 = np.sqrt((0.5-x)**2 + (0.5-y)**2)
    theta1 = np.arctan2(0.5-x, 0.5-y)
    r2 = np.sqrt((-x-0.2)**2 + (-y-0.2)**2)
    theta2 = np.arctan2(-x-0.2, -y-0.2)
    z = -(2*(np.exp((r1/10)**2)-1)*30. * np.cos(7.*theta1) +
          (np.exp((r2/10)**2)-1)*30. * np.cos(11.*theta2) +
          0.7*(x**2 + y**2))
    return (np.max(z)-z)/(np.max(z)-np.min(z))

#-----------------------------------------------------------------------------
# Creating a Triangulation
#-----------------------------------------------------------------------------
# First create the x and y coordinates of the points.
n_angles = 20
n_radii = 10
min_radius = 0.15
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
z = function_z(x, y)

# Now create the Triangulation.
# (Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.)
triang = Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

#-----------------------------------------------------------------------------
# Refine data
#-----------------------------------------------------------------------------
refiner = UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)

#-----------------------------------------------------------------------------
# Plot the triangulation and the high-res iso-contours
#-----------------------------------------------------------------------------
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(triang, lw=0.5, color='white')

levels = np.arange(0., 1., 0.025)
cmap = cm.get_cmap(name='terrain', lut=None)
plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
plt.tricontour(tri_refi, z_test_refi, levels=levels,
               colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
               linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])

plt.title("High-resolution tricontouring")

plt.show()




********************************************
./pylab_examples/csd_demo.py
#!/usr/bin/env python
"""
Compute the cross spectral density of two signals
"""
import numpy as np
import matplotlib.pyplot as plt

# make a little extra space between the subplots
plt.subplots_adjust(wspace=0.5)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2
r = np.exp(-t/0.05)

cnse1 = np.convolve(nse1, r, mode='same')*dt   # colored noise 1
cnse2 = np.convolve(nse2, r, mode='same')*dt   # colored noise 2

# two signals with a coherent part and a random part
s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1
s2 = 0.01*np.sin(2*np.pi*10*t) + cnse2

plt.subplot(211)
plt.plot(t, s1, 'b-', t, s2, 'g-')
plt.xlim(0,5)
plt.xlabel('time')
plt.ylabel('s1 and s2')
plt.grid(True)

plt.subplot(212)
cxy, f = plt.csd(s1, s2, 256, 1./dt)
plt.ylabel('CSD (db)')
plt.show()






********************************************
./pylab_examples/finance_work2.py
import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


startdate = datetime.date(2006,1,1)
today = enddate = datetime.date.today()
ticker = 'SPY'


fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
# a numpy record array with fields: date, open, high, low, close, volume, adj_close)

r = mlab.csv2rec(fh); fh.close()
r.sort()


def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()


    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow


plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

textsize = 9
left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]


fig = plt.figure(facecolor='white')
axescolor  = '#f6f6f6'  # the axes background color

ax1 = fig.add_axes(rect1, axisbg=axescolor)  #left, bottom, width, height
ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
ax2t = ax2.twinx()
ax3  = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)



### plot the relative strength indicator
prices = r.adj_close
rsi = relative_strength(prices)
fillcolor = 'darkgoldenrod'

ax1.plot(r.date, rsi, color=fillcolor)
ax1.axhline(70, color=fillcolor)
ax1.axhline(30, color=fillcolor)
ax1.fill_between(r.date, rsi, 70, where=(rsi>=70), facecolor=fillcolor, edgecolor=fillcolor)
ax1.fill_between(r.date, rsi, 30, where=(rsi<=30), facecolor=fillcolor, edgecolor=fillcolor)
ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
ax1.set_ylim(0, 100)
ax1.set_yticks([30,70])
ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.set_title('%s daily'%ticker)

### plot the price and volume data
dx = r.adj_close - r.close
low = r.low + dx
high = r.high + dx

deltas = np.zeros_like(prices)
deltas[1:] = np.diff(prices)
up = deltas>0
ax2.vlines(r.date[up], low[up], high[up], color='black', label='_nolegend_')
ax2.vlines(r.date[~up], low[~up], high[~up], color='black', label='_nolegend_')
ma20 = moving_average(prices, 20, type='simple')
ma200 = moving_average(prices, 200, type='simple')

linema20, = ax2.plot(r.date, ma20, color='blue', lw=2, label='MA (20)')
linema200, = ax2.plot(r.date, ma200, color='red', lw=2, label='MA (200)')


last = r[-1]
s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
    today.strftime('%d-%b-%Y'),
    last.open, last.high,
    last.low, last.close,
    last.volume*1e-6,
    last.close-last.open )
t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

props = font_manager.FontProperties(size=10)
leg = ax2.legend(loc='center left', shadow=True, fancybox=True, prop=props)
leg.get_frame().set_alpha(0.5)


volume = (r.close*r.volume)/1e6  # dollar volume in millions
vmax = volume.max()
poly = ax2t.fill_between(r.date, volume, 0, label='Volume', facecolor=fillcolor, edgecolor=fillcolor)
ax2t.set_ylim(0, 5*vmax)
ax2t.set_yticks([])


### compute the MACD indicator
fillcolor = 'darkslategrey'
nslow = 26
nfast = 12
nema = 9
emaslow, emafast, macd = moving_average_convergence(prices, nslow=nslow, nfast=nfast)
ema9 = moving_average(macd, nema, type='exponential')
ax3.plot(r.date, macd, color='black', lw=2)
ax3.plot(r.date, ema9, color='blue', lw=1)
ax3.fill_between(r.date, macd-ema9, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)


ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)'%(nfast, nslow, nema), va='top',
         transform=ax3.transAxes, fontsize=textsize)

#ax3.set_yticks([])
# turn off upper axis tick labels, rotate the lower ones, etc
for ax in ax1, ax2, ax2t, ax3:
    if ax!=ax3:
        for label in ax.get_xticklabels():
            label.set_visible(False)
    else:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')



class MyLocator(mticker.MaxNLocator):
    def __init__(self, *args, **kwargs):
        mticker.MaxNLocator.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return mticker.MaxNLocator.__call__(self, *args, **kwargs)

# at most 5 ticks, pruning the upper and lower so they don't overlap
# with other ticks
#ax2.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
#ax3.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))

ax2.yaxis.set_major_locator(MyLocator(5, prune='both'))
ax3.yaxis.set_major_locator(MyLocator(5, prune='both'))

plt.show()






********************************************
./pylab_examples/scatter_masked.py
#!/usr/bin/env python
from pylab import *

N = 100
r0 = 0.6
x = 0.9*rand(N)
y = 0.9*rand(N)
area = pi*(10 * rand(N))**2 # 0 to 10 point radiuses
c = sqrt(area)
r = sqrt(x*x+y*y)
area1 = ma.masked_where(r < r0, area)
area2 = ma.masked_where(r >= r0, area)
scatter(x,y,s=area1, marker='^', c=c, hold='on')
scatter(x,y,s=area2, marker='o', c=c)
# Show the boundary between the regions:
theta = arange(0, pi/2, 0.01)
plot(r0*cos(theta), r0*sin(theta))

show()




********************************************
./pylab_examples/hyperlinks.py
#!/usr/bin/env python
# -*- noplot -*-

"""
This example demonstrates how to set a hyperlinks on various kinds of elements.

This currently only works with the SVG backend.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

f = plt.figure()
s = plt.scatter([1,2,3],[4,5,6])
s.set_urls(['http://www.bbc.co.uk/news','http://www.google.com',None])
f.canvas.print_figure('scatter.svg')

f = plt.figure()
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                origin='lower', extent=[-3,3,-3,3])

im.set_url('http://www.google.com')
f.canvas.print_figure('image.svg')





********************************************
./pylab_examples/color_by_yvalue.py
# use masked arrays to plot a line with different colors by y-value
from numpy import logical_or, arange, sin, pi
from numpy import ma
from matplotlib.pyplot import  plot, show

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)

upper = 0.77
lower = -0.77


supper = ma.masked_where(s < upper, s)
slower = ma.masked_where(s > lower, s)
smiddle = ma.masked_where(logical_or(s<lower, s>upper), s)

plot(t, slower, 'r', t, smiddle, 'b', t, supper, 'g')
show()




********************************************
./pylab_examples/step_demo.py
import numpy as np
from numpy import ma
from matplotlib.pyplot import step, legend, xlim, ylim, show

x = np.arange(1, 7, 0.4)
y0 = np.sin(x)
y = y0.copy() + 2.5

step(x, y, label='pre (default)')

y -= 0.5
step(x, y, where='mid', label='mid')

y -= 0.5
step(x, y, where='post', label='post')

y = ma.masked_where((y0>-0.15)&(y0<0.15), y - 0.5)
step(x,y, label='masked (pre)')

legend()

xlim(0, 7)
ylim(-0.5, 4)

show()





********************************************
./pylab_examples/psd_demo2.py
#This example shows the effects of some of the different PSD parameters
import numpy as np
import matplotlib.pyplot as plt

dt = np.pi / 100.
fs = 1. / dt
t = np.arange(0, 8, dt)
y = 10. * np.sin(2 * np.pi * 4 * t) + 5. * np.sin(2 * np.pi * 4.25 * t)
y = y + np.random.randn(*t.shape)

#Plot the raw time series
fig = plt.figure()
fig.subplots_adjust(hspace=0.45, wspace=0.3)
ax = fig.add_subplot(2, 1, 1)
ax.plot(t, y)

#Plot the PSD with different amounts of zero padding. This uses the entire
#time series at once
ax2 = fig.add_subplot(2, 3, 4)
ax2.psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
ax2.psd(y, NFFT=len(t), pad_to=len(t)*2, Fs=fs)
ax2.psd(y, NFFT=len(t), pad_to=len(t)*4, Fs=fs)
plt.title('zero padding')

#Plot the PSD with different block sizes, Zero pad to the length of the original
#data sequence.
ax3 = fig.add_subplot(2, 3, 5, sharex=ax2, sharey=ax2)
ax3.psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
ax3.psd(y, NFFT=len(t)//2, pad_to=len(t), Fs=fs)
ax3.psd(y, NFFT=len(t)//4, pad_to=len(t), Fs=fs)
ax3.set_ylabel('')
plt.title('block size')

#Plot the PSD with different amounts of overlap between blocks
ax4 = fig.add_subplot(2, 3, 6, sharex=ax2, sharey=ax2)
ax4.psd(y, NFFT=len(t)//2, pad_to=len(t), noverlap=0, Fs=fs)
ax4.psd(y, NFFT=len(t)//2, pad_to=len(t), noverlap=int(0.05*len(t)/2.), Fs=fs)
ax4.psd(y, NFFT=len(t)//2, pad_to=len(t), noverlap=int(0.2*len(t)/2.), Fs=fs)
ax4.set_ylabel('')
plt.title('overlap')

plt.show()




********************************************
./pylab_examples/demo_text_path.py

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
import numpy as np
from matplotlib.transforms import IdentityTransform

import matplotlib.patches as mpatches

from matplotlib.offsetbox import AnnotationBbox,\
     AnchoredOffsetbox, AuxTransformBox

from matplotlib.cbook import get_sample_data

from matplotlib.text import TextPath


class PathClippedImagePatch(mpatches.PathPatch):
    """
    The given image is used to draw the face of the patch. Internally,
    it uses BboxImage whose clippath set to the path of the patch.

    FIXME : The result is currently dpi dependent.
    """
    def __init__(self, path, bbox_image, **kwargs):
        mpatches.PathPatch.__init__(self, path, **kwargs)
        self._init_bbox_image(bbox_image)

    def set_facecolor(self, color):
        """simply ignore facecolor"""
        mpatches.PathPatch.set_facecolor(self, "none")
    
    def _init_bbox_image(self, im):

        bbox_image = BboxImage(self.get_window_extent,
                               norm = None,
                               origin=None,
                               )
        bbox_image.set_transform(IdentityTransform())

        bbox_image.set_data(im)
        self.bbox_image = bbox_image

    def draw(self, renderer=None):


        # the clip path must be updated every draw. any solution? -JJ
        self.bbox_image.set_clip_path(self._path, self.get_transform())
        self.bbox_image.draw(renderer)

        mpatches.PathPatch.draw(self, renderer)


if 1:

    usetex = plt.rcParams["text.usetex"]
    
    fig = plt.figure(1)

    # EXAMPLE 1

    ax = plt.subplot(211)

    from matplotlib._png import read_png
    fn = get_sample_data("grace_hopper.png", asfileobj=False)
    arr = read_png(fn)

    text_path = TextPath((0, 0), "!?", size=150)
    p = PathClippedImagePatch(text_path, arr, ec="k",
                              transform=IdentityTransform())

    #p.set_clip_on(False)

    # make offset box
    offsetbox = AuxTransformBox(IdentityTransform())
    offsetbox.add_artist(p)

    # make anchored offset box
    ao = AnchoredOffsetbox(loc=2, child=offsetbox, frameon=True, borderpad=0.2)
    ax.add_artist(ao)

    # another text
    from matplotlib.patches import PathPatch
    if usetex:
        r = r"\mbox{textpath supports mathtext \& \TeX}"
    else:
        r = r"textpath supports mathtext & TeX"
        
    text_path = TextPath((0, 0), r,
                         size=20, usetex=usetex)
        
    p1 = PathPatch(text_path, ec="w", lw=3, fc="w", alpha=0.9,
                   transform=IdentityTransform())
    p2 = PathPatch(text_path, ec="none", fc="k", 
                   transform=IdentityTransform())

    offsetbox2 = AuxTransformBox(IdentityTransform())
    offsetbox2.add_artist(p1)
    offsetbox2.add_artist(p2)

    ab = AnnotationBbox(offsetbox2, (0.95, 0.05),
                        xycoords='axes fraction',
                        boxcoords="offset points",
                        box_alignment=(1.,0.),
                        frameon=False
                        )
    ax.add_artist(ab)

    ax.imshow([[0,1,2],[1,2,3]], cmap=plt.cm.gist_gray_r,
              interpolation="bilinear",
              aspect="auto")



    # EXAMPLE 2

    ax = plt.subplot(212)

    arr = np.arange(256).reshape(1,256)/256.

    if usetex:
        s = r"$\displaystyle\left[\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}\right]$!"
    else:
        s = r"$\left[\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}\right]$!"
    text_path = TextPath((0, 0), s, size=40, usetex=usetex)
    text_patch = PathClippedImagePatch(text_path, arr, ec="none",
                                       transform=IdentityTransform())

    shadow1 = mpatches.Shadow(text_patch, 1, -1, props=dict(fc="none", ec="0.6", lw=3))
    shadow2 = mpatches.Shadow(text_patch, 1, -1, props=dict(fc="0.3", ec="none"))


    # make offset box
    offsetbox = AuxTransformBox(IdentityTransform())
    offsetbox.add_artist(shadow1)
    offsetbox.add_artist(shadow2)
    offsetbox.add_artist(text_patch)

    # place the anchored offset box using AnnotationBbox
    ab = AnnotationBbox(offsetbox, (0.5, 0.5),
                        xycoords='data',
                        boxcoords="offset points",
                        box_alignment=(0.5,0.5),
                        )
    #text_path.set_size(10)

    ax.add_artist(ab)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    plt.draw()
    plt.show()




********************************************
./pylab_examples/text_handles.py
#!/usr/bin/env python
#Controlling the properties of axis text using handles

# See examples/text_themes.py for a more elegant, pythonic way to control
# fonts.  After all, if we were slaves to MATLAB , we wouldn't be
# using python!

from pylab import *


def f(t):
    s1 = sin(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)


fig, ax = plt.subplots()
plot(t1, f(t1), 'bo', t2, f(t2), 'k')
text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
ttext = title('Fun with text!')
ytext = ylabel('Damped oscillation')
xtext = xlabel('time (s)')

setp(ttext, size='large', color='r', style='italic')
setp(xtext, size='medium', name='courier', weight='bold', color='g')
setp(ytext, size='medium', name='helvetica', weight='light', color='b')
show()




********************************************
./pylab_examples/fonts_demo_kw.py
#!/usr/bin/env python
"""
Same as fonts_demo using kwargs.  If you prefer a more pythonic, OO
style of coding, see examples/fonts_demo.py.

"""
from matplotlib.font_manager import FontProperties
from pylab import *

subplot(111, axisbg='w')
alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
###  Show family options

family = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

t = text(-0.8, 0.9, 'family', size='large', **alignment)

yp = [0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5]

for k in range(5):
    if k == 2:
        t = text(-0.8, yp[k], family[k], family=family[k],
                 name='Script MT', **alignment)
    else:
        t = text(-0.8, yp[k], family[k], family=family[k], **alignment)

###  Show style options

style  = ['normal', 'italic', 'oblique']

t = text(-0.4, 0.9, 'style', **alignment)

for k in range(3):
    t = text(-0.4, yp[k], style[k], family='sans-serif', style=style[k],
             **alignment)

###  Show variant options

variant= ['normal', 'small-caps']

t = text(0.0, 0.9, 'variant', **alignment)

for k in range(2):
    t = text( 0.0, yp[k], variant[k], family='serif', variant=variant[k],
              **alignment)

###  Show weight options

weight = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = text( 0.4, 0.9, 'weight',  **alignment)

for k in range(7):
    t = text( 0.4, yp[k], weight[k], weight=weight[k],
              **alignment)

###  Show size options

size  = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = text( 0.8, 0.9, 'size', **alignment)

for k in range(7):
    t = text( 0.8, yp[k], size[k], size=size[k],
             **alignment)

x = 0
###  Show bold italic
t = text(x, 0.1, 'bold italic', style='italic',
         weight='bold', size='x-small',
         **alignment)

t = text(x, 0.2, 'bold italic',
         style = 'italic', weight='bold', size='medium',
         **alignment)

t = text(x, 0.3, 'bold italic',
         style='italic', weight='bold', size='x-large',
         **alignment)

axis([-1, 1, 0, 1])

show()




********************************************
./pylab_examples/print_stdout.py
#!/usr/bin/env python
# -*- noplot -*-
# print png to standard out
# usage: python print_stdout.py > somefile.png
import sys
import matplotlib
matplotlib.use('Agg')
from pylab import *

plot([1,2,3])

savefig(sys.stdout)
show()




********************************************
./pylab_examples/colorbar_tick_labelling_demo.py
"""Produce custom labelling for a colorbar.

Contributed by Scott Sinclair
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import randn

# Make plot with vertical (default) colorbar
fig, ax = plt.subplots()

data = np.clip(randn(250, 250), -1, 1)

cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Gaussian noise with vertical colorbar')

# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

# Make plot with horizontal colorbar
fig, ax = plt.subplots()

data = np.clip(randn(250, 250), -1, 1)

cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot)
ax.set_title('Gaussian noise with horizontal colorbar')

cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])# horizontal colorbar

plt.show()




********************************************
./pylab_examples/demo_agg_filter.py
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab

def smooth1d(x, window_len):
    # copied from http://www.scipy.org/Cookbook/SignalSmooth

    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def smooth2d(A, sigma=3):

    window_len = max(int(sigma), 3)*2+1
    A1 = np.array([smooth1d(x, window_len) for x in np.asarray(A)])
    A2 = np.transpose(A1)
    A3 = np.array([smooth1d(x, window_len) for x in A2])
    A4 = np.transpose(A3)

    return A4




class BaseFilter(object):
    def prepare_image(self, src_image, dpi, pad):
        ny, nx, depth = src_image.shape
        #tgt_image = np.zeros([pad*2+ny, pad*2+nx, depth], dtype="d")
        padded_src = np.zeros([pad*2+ny, pad*2+nx, depth], dtype="d")
        padded_src[pad:-pad, pad:-pad,:] = src_image[:,:,:]

        return padded_src#, tgt_image

    def get_pad(self, dpi):
        return 0

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = self.prepare_image(im, dpi, pad)
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


class OffsetFilter(BaseFilter):
    def __init__(self, offsets=None):
        if offsets is None:
            self.offsets = (0, 0)
        else:
            self.offsets = offsets

    def get_pad(self, dpi):
        return int(max(*self.offsets)/72.*dpi)

    def process_image(self, padded_src, dpi):
        ox, oy = self.offsets
        a1 = np.roll(padded_src, int(ox/72.*dpi), axis=1)
        a2 = np.roll(a1, -int(oy/72.*dpi), axis=0)
        return a2

class GaussianFilter(BaseFilter):
    "simple gauss filter"
    def __init__(self, sigma, alpha=0.5, color=None):
        self.sigma = sigma
        self.alpha = alpha
        if color is None:
            self.color=(0, 0, 0)
        else:
            self.color=color

    def get_pad(self, dpi):
        return int(self.sigma*3/72.*dpi)


    def process_image(self, padded_src, dpi):
        #offsetx, offsety = int(self.offsets[0]), int(self.offsets[1])
        tgt_image = np.zeros_like(padded_src)
        aa = smooth2d(padded_src[:,:,-1]*self.alpha,
                      self.sigma/72.*dpi)
        tgt_image[:,:,-1] = aa
        tgt_image[:,:,:-1] = self.color
        return tgt_image

class DropShadowFilter(BaseFilter):
    def __init__(self, sigma, alpha=0.3, color=None, offsets=None):
        self.gauss_filter = GaussianFilter(sigma, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi),
                   self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        t2 = self.offset_filter.process_image(t1, dpi)
        return t2


from matplotlib.colors import LightSource

class LightFilter(BaseFilter):
    "simple gauss filter"
    def __init__(self, sigma, fraction=0.5):
        self.gauss_filter = GaussianFilter(sigma, alpha=1)
        self.light_source = LightSource()
        self.fraction = fraction
        #hsv_min_val=0.5,hsv_max_val=0.9,
        #                                hsv_min_sat=0.1,hsv_max_sat=0.1)
    def get_pad(self, dpi):
        return self.gauss_filter.get_pad(dpi)

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        elevation = t1[:,:,3]
        rgb = padded_src[:,:,:3]

        rgb2 = self.light_source.shade_rgb(rgb, elevation,
                                           fraction=self.fraction)

        tgt = np.empty_like(padded_src)
        tgt[:,:,:3] = rgb2
        tgt[:,:,3] = padded_src[:,:,3]

        return tgt



class GrowFilter(BaseFilter):
    "enlarge the area"
    def __init__(self, pixels, color=None):
        self.pixels = pixels
        if color is None:
            self.color=(1, 1, 1)
        else:
            self.color=color

    def __call__(self, im, dpi):
        pad = self.pixels
        ny, nx, depth = im.shape
        new_im = np.empty([pad*2+ny, pad*2+nx, depth], dtype="d")
        alpha = new_im[:,:,3]
        alpha.fill(0)
        alpha[pad:-pad, pad:-pad] = im[:,:,-1]
        alpha2 = np.clip(smooth2d(alpha, self.pixels/72.*dpi) * 5, 0, 1)
        new_im[:,:,-1] = alpha2
        new_im[:,:,:-1] = self.color
        offsetx, offsety = -pad, -pad

        return new_im, offsetx, offsety


from matplotlib.artist import Artist

class FilteredArtistList(Artist):
    """
    A simple container to draw filtered artist.
    """
    def __init__(self, artist_list, filter):
        self._artist_list = artist_list
        self._filter = filter
        Artist.__init__(self)

    def draw(self, renderer):
        renderer.start_rasterizing()
        renderer.start_filter()
        for a in self._artist_list:
            a.draw(renderer)
        renderer.stop_filter(self._filter)
        renderer.stop_rasterizing()



import matplotlib.transforms as mtransforms

def filtered_text(ax):
    # mostly copied from contour_demo.py

    # prepare image
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)


    # draw
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                   cmap=cm.gray, extent=(-3,3,-2,2))
    levels = np.arange(-1.2, 1.6, 0.2)
    CS = ax.contour(Z, levels,
                    origin='lower',
                    linewidths=2,
                    extent=(-3,3,-2,2))

    ax.set_aspect("auto")

    # contour label
    cl = ax.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%1.1f',
                   fontsize=11)

    # change clable color to black
    from matplotlib.patheffects import Normal
    for t in cl:
        t.set_color("k")
        t.set_path_effects([Normal()]) # to force TextPath (i.e., same font in all backends)

    # Add white glows to improve visibility of labels.
    white_glows = FilteredArtistList(cl, GrowFilter(3))
    ax.add_artist(white_glows)
    white_glows.set_zorder(cl[0].get_zorder()-0.1)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def drop_shadow_line(ax):
    # copied from examples/misc/svg_filter_line.py

    # draw lines
    l1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-",
                  mec="b", mfc="w", lw=5, mew=3, ms=10, label="Line 1")
    l2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "ro-",
                  mec="r", mfc="w", lw=5, mew=3, ms=10, label="Line 1")


    gauss = DropShadowFilter(4)

    for l in [l1, l2]:

        # draw shadows with same lines with slight offset.

        xx = l.get_xdata()
        yy = l.get_ydata()
        shadow, = ax.plot(xx, yy)
        shadow.update_from(l)

        # offset transform
        ot = mtransforms.offset_copy(l.get_transform(), ax.figure,
                                     x=4.0, y=-6.0, units='points')

        shadow.set_transform(ot)


        # adjust zorder of the shadow lines so that it is drawn below the
        # original lines
        shadow.set_zorder(l.get_zorder()-0.5)
        shadow.set_agg_filter(gauss)
        shadow.set_rasterized(True) # to support mixed-mode renderers



    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)




def drop_shadow_patches(ax):
    # copyed from barchart_demo.py
    N = 5
    menMeans = (20, 35, 30, 35, 27)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    rects1 = ax.bar(ind, menMeans, width, color='r', ec="w", lw=2)

    womenMeans = (25, 32, 34, 20, 25)
    rects2 = ax.bar(ind+width+0.1, womenMeans, width, color='y', ec="w", lw=2)

    #gauss = GaussianFilter(1.5, offsets=(1,1), )
    gauss = DropShadowFilter(5, offsets=(1,1), )
    shadow = FilteredArtistList(rects1+rects2, gauss)
    ax.add_artist(shadow)
    shadow.set_zorder(rects1[0].get_zorder()-0.1)

    ax.set_xlim(ind[0]-0.5, ind[-1]+1.5)
    ax.set_ylim(0, 40)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def light_filter_pie(ax):
    fracs = [15,30,45, 10]
    explode=(0, 0.05, 0, 0)
    pies = ax.pie(fracs, explode=explode)
    ax.patch.set_visible(True)

    light_filter = LightFilter(9)
    for p in pies[0]:
        p.set_agg_filter(light_filter)
        p.set_rasterized(True) # to support mixed-mode renderers
        p.set(ec="none",
              lw=2)

    gauss = DropShadowFilter(9, offsets=(3,4), alpha=0.7)
    shadow = FilteredArtistList(pies[0], gauss)
    ax.add_artist(shadow)
    shadow.set_zorder(pies[0][0].get_zorder()-0.1)


if 1:
 
    plt.figure(1, figsize=(6, 6))
    plt.subplots_adjust(left=0.05, right=0.95)

    ax = plt.subplot(221)
    filtered_text(ax)

    ax = plt.subplot(222)
    drop_shadow_line(ax)

    ax = plt.subplot(223)
    drop_shadow_patches(ax)

    ax = plt.subplot(224)
    ax.set_aspect(1)
    light_filter_pie(ax)
    ax.set_frame_on(True)

    plt.show()






********************************************
./pylab_examples/font_table_ttf.py
#!/usr/bin/env python
# -*- noplot -*-
"""
matplotlib has support for freetype fonts.  Here's a little example
using the 'table' command to build a font table that shows the glyphs
by character code.

Usage python font_table_ttf.py somefile.ttf
"""

import sys
import os

import matplotlib
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import FontProperties
from pylab import figure, table, show, axis, title

try:
    unichr
except NameError:
    # Python 3
    unichr = chr

# the font table grid

labelc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F']
labelr = ['00', '10', '20', '30', '40', '50', '60', '70', '80', '90',
          'A0', 'B0', 'C0', 'D0', 'E0', 'F0']

if len(sys.argv) > 1:
    fontname = sys.argv[1]
else:
    fontname = os.path.join(matplotlib.get_data_path(),
                            'fonts', 'ttf', 'Vera.ttf')

font = FT2Font(fontname)
codes = list(font.get_charmap().items())
codes.sort()

# a 16,16 array of character strings
chars = [['' for c in range(16)] for r in range(16)]
colors = [[(0.95, 0.95, 0.95) for c in range(16)] for r in range(16)]

figure(figsize=(8, 4), dpi=120)
for ccode, glyphind in codes:
    if ccode >= 256:
        continue
    r, c = divmod(ccode, 16)
    s = unichr(ccode)
    chars[r][c] = s

lightgrn = (0.5, 0.8, 0.5)
title(fontname)
tab = table(cellText=chars,
            rowLabels=labelr,
            colLabels=labelc,
            rowColours=[lightgrn]*16,
            colColours=[lightgrn]*16,
            cellColours=colors,
            cellLoc='center',
            loc='upper left')

for key, cell in tab.get_celld().items():
    row, col = key
    if row > 0 and col > 0:
        cell.set_text_props(fontproperties=FontProperties(fname=fontname))
axis('off')
show()




********************************************
./pylab_examples/date_demo2.py
#!/usr/bin/env python

"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks
"""
from __future__ import print_function
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import MONDAY
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter


date1 = datetime.date(2002, 1, 5)
date2 = datetime.date(2003, 12, 1)

# every monday
mondays = WeekdayLocator(MONDAY)

# every 3rd month
months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
monthsFmt = DateFormatter("%b '%y")


quotes = quotes_historical_yahoo_ochl('INTC', date1, date2)
if len(quotes) == 0:
    print ('Found no quotes')
    raise SystemExit

dates = [q[0] for q in quotes]
opens = [q[1] for q in quotes]

fig, ax = plt.subplots()
ax.plot_date(dates, opens, '-')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(mondays)
ax.autoscale_view()
#ax.xaxis.grid(False, 'major')
#ax.xaxis.grid(True, 'minor')
ax.grid(True)

fig.autofmt_xdate()

plt.show()




********************************************
./pylab_examples/legend_demo4.py
import matplotlib.pyplot as plt

ax = plt.subplot(311)

b1 = ax.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4,
            label="Bar 1", align="center")

b2 = ax.bar([0.5, 1.5, 2.5], [0.3, 0.2, 0.2], color="red", width=0.4,
            label="Bar 2", align="center")

ax.legend()

ax = plt.subplot(312)

err1 = ax.errorbar([0, 1, 2], [2, 3, 1], xerr=0.4, fmt="s",
                   label="test 1")
err2 = ax.errorbar([0, 1, 2], [3, 2, 4], yerr=0.3, fmt="o",
                   label="test 2")
err3 = ax.errorbar([0, 1, 2], [1, 1, 3], xerr=0.4, yerr=0.3, fmt="^",
                   label="test 3")

ax.legend()

ax = plt.subplot(313)

ll = ax.stem([0.3, 1.5, 2.7], [1, 3.6, 2.7], label="stem test")

ax.legend()

plt.show()





********************************************
./pylab_examples/tricontour_demo.py
"""
Contour plots of unstructured triangular grids.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math

# Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.

# First create the x and y coordinates of the points.
n_angles = 48
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi/n_angles

x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
z = (np.cos(radii)*np.cos(angles*3.0)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

# pcolor plot.
plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(triang, z)
plt.colorbar()
plt.tricontour(triang, z, colors='k')
plt.title('Contour plot of Delaunay triangulation')

# You can specify your own triangulation rather than perform a Delaunay
# triangulation of the points, where each triangle is given by the indices of
# the three points that make up the triangle, ordered in either a clockwise or
# anticlockwise manner.

xy = np.asarray([
    [-0.101, 0.872], [-0.080, 0.883], [-0.069, 0.888], [-0.054, 0.890],
    [-0.045, 0.897], [-0.057, 0.895], [-0.073, 0.900], [-0.087, 0.898],
    [-0.090, 0.904], [-0.069, 0.907], [-0.069, 0.921], [-0.080, 0.919],
    [-0.073, 0.928], [-0.052, 0.930], [-0.048, 0.942], [-0.062, 0.949],
    [-0.054, 0.958], [-0.069, 0.954], [-0.087, 0.952], [-0.087, 0.959],
    [-0.080, 0.966], [-0.085, 0.973], [-0.087, 0.965], [-0.097, 0.965],
    [-0.097, 0.975], [-0.092, 0.984], [-0.101, 0.980], [-0.108, 0.980],
    [-0.104, 0.987], [-0.102, 0.993], [-0.115, 1.001], [-0.099, 0.996],
    [-0.101, 1.007], [-0.090, 1.010], [-0.087, 1.021], [-0.069, 1.021],
    [-0.052, 1.022], [-0.052, 1.017], [-0.069, 1.010], [-0.064, 1.005],
    [-0.048, 1.005], [-0.031, 1.005], [-0.031, 0.996], [-0.040, 0.987],
    [-0.045, 0.980], [-0.052, 0.975], [-0.040, 0.973], [-0.026, 0.968],
    [-0.020, 0.954], [-0.006, 0.947], [ 0.003, 0.935], [ 0.006, 0.926],
    [ 0.005, 0.921], [ 0.022, 0.923], [ 0.033, 0.912], [ 0.029, 0.905],
    [ 0.017, 0.900], [ 0.012, 0.895], [ 0.027, 0.893], [ 0.019, 0.886],
    [ 0.001, 0.883], [-0.012, 0.884], [-0.029, 0.883], [-0.038, 0.879],
    [-0.057, 0.881], [-0.062, 0.876], [-0.078, 0.876], [-0.087, 0.872],
    [-0.030, 0.907], [-0.007, 0.905], [-0.057, 0.916], [-0.025, 0.933],
    [-0.077, 0.990], [-0.059, 0.993]])
x = xy[:, 0]*180/3.14159
y = xy[:, 1]*180/3.14159
x0 = -5
y0 = 52
z = np.exp(-0.01*((x-x0)*(x-x0) + (y-y0)*(y-y0)))

triangles = np.asarray([
    [67, 66,  1], [65,  2, 66], [ 1, 66,  2], [64,  2, 65], [63,  3, 64],
    [60, 59, 57], [ 2, 64,  3], [ 3, 63,  4], [ 0, 67,  1], [62,  4, 63],
    [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [ 4, 62, 68],
    [ 6,  5,  9], [61, 68, 62], [69, 68, 61], [ 9,  5, 70], [ 6,  8,  7],
    [ 4, 70,  5], [ 8,  6,  9], [56, 69, 57], [69, 56, 52], [70, 10,  9],
    [54, 53, 55], [56, 55, 53], [68, 70,  4], [52, 56, 53], [11, 10, 12],
    [69, 71, 68], [68, 13, 70], [10, 70, 13], [51, 50, 52], [13, 68, 71],
    [52, 71, 69], [12, 10, 13], [71, 52, 50], [71, 14, 13], [50, 49, 71],
    [49, 48, 71], [14, 16, 15], [14, 71, 48], [17, 19, 18], [17, 20, 19],
    [48, 16, 14], [48, 47, 16], [47, 46, 16], [16, 46, 45], [23, 22, 24],
    [21, 24, 22], [17, 16, 45], [20, 17, 45], [21, 25, 24], [27, 26, 28],
    [20, 72, 21], [25, 21, 72], [45, 72, 20], [25, 28, 26], [44, 73, 45],
    [72, 45, 73], [28, 25, 29], [29, 25, 31], [43, 73, 44], [73, 43, 40],
    [72, 73, 39], [72, 31, 25], [42, 40, 43], [31, 30, 29], [39, 73, 40],
    [42, 41, 40], [72, 33, 31], [32, 31, 33], [39, 38, 72], [33, 72, 38],
    [33, 38, 34], [37, 35, 38], [34, 38, 35], [35, 37, 36]])

# Rather than create a Triangulation object, can simply pass x, y and triangles
# arrays to tripcolor directly.  It would be better to use a Triangulation
# object if the same triangulation was to be used more than once to save
# duplicated calculations.
plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(x, y, triangles, z)
plt.colorbar()
plt.title('Contour plot of user-specified triangulation')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')

plt.show()




********************************************
./pylab_examples/hist2d_demo.py
from pylab import *
x = randn(1000)
y = randn(1000)+5

#normal distribution center at x=0 and y=5
hist2d(x,y,bins=40)
show()




********************************************
./pylab_examples/legend_auto.py
"""
This file was written to test matplotlib's autolegend placement
algorithm, but shows lots of different ways to create legends so is
useful as a general examples

Thanks to John Gill and Phil ?? for help at the matplotlib sprint at
pycon 2005 where the auto-legend support was written.
"""
from pylab import *
import sys

rcParams['legend.loc'] = 'best'

N = 100
x = arange(N)

def fig_1():
    figure(1)
    t = arange(0, 40.0 * pi, 0.1)
    l, = plot(t, 100*sin(t), 'r', label='sine')
    legend(framealpha=0.5)

def fig_2():
    figure(2)
    plot(x, 'o', label='x=y')
    legend()

def fig_3():
    figure(3)
    plot(x, -x, 'o', label='x= -y')
    legend()

def fig_4():
    figure(4)
    plot(x, ones(len(x)), 'o', label='y=1')
    plot(x, -ones(len(x)), 'o', label='y=-1')
    legend()

def fig_5():
    figure(5)
    n, bins, patches = hist(randn(1000), 40, normed=1)
    l, = plot(bins, normpdf(bins, 0.0, 1.0), 'r--', label='fit', linewidth=3)
    legend([l, patches[0]], ['fit', 'hist'])

def fig_6():
    figure(6)
    plot(x, 50-x, 'o', label='y=1')
    plot(x, x-50, 'o', label='y=-1')
    legend()

def fig_7():
    figure(7)
    xx = x - (N/2.0)
    plot(xx, (xx*xx)-1225, 'bo', label='$y=x^2$')
    plot(xx, 25*xx, 'go', label='$y=25x$')
    plot(xx, -25*xx, 'mo', label='$y=-25x$')
    legend()

def fig_8():
    figure(8)
    b1 = bar(x, x, color='m')
    b2 = bar(x, x[::-1], color='g')
    legend([b1[0], b2[0]], ['up', 'down'])

def fig_9():
    figure(9)
    b1 = bar(x, -x)
    b2 = bar(x, -x[::-1], color='r')
    legend([b1[0], b2[0]], ['down', 'up'])

def fig_10():
    figure(10)
    b1 = bar(x, x, bottom=-100, color='m')
    b2 = bar(x, x[::-1], bottom=-100, color='g')
    b3 = bar(x, -x, bottom=100)
    b4 = bar(x, -x[::-1], bottom=100, color='r')
    legend([b1[0], b2[0], b3[0], b4[0]], ['bottom right', 'bottom left',
                                          'top left', 'top right'])

if __name__ == '__main__':
    nfigs = 10
    figures = []
    for f in sys.argv[1:]:
        try:
            figures.append(int(f))
        except ValueError:
            pass
    if len(figures) == 0:
        figures = range(1, nfigs+1)

    for fig in figures:
        fn_name = "fig_%d" % fig
        fn = globals()[fn_name]
        fn()

    show()




********************************************
./pylab_examples/ellipse_rotated.py
from pylab import *
from matplotlib.patches import Ellipse

delta = 45.0 # degrees

angles = arange(0, 360+delta, delta)
ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

a = subplot(111, aspect='equal')

for e in ells:
    e.set_clip_box(a.bbox)
    e.set_alpha(0.1)
    a.add_artist(e)

xlim(-2, 4)
ylim(-1, 3)

show()




********************************************
./pylab_examples/image_demo.py
#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians

im = plt.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                origin='lower', extent=[-3,3,-3,3],
                vmax=abs(Z).max(), vmin=-abs(Z).max())

plt.show()





********************************************
./pylab_examples/break.py
#!/usr/bin/env python
from pylab import *


gcf().text(0.5, 0.95,
           'Distance Histograms by Category is a really long title')

show()




********************************************
./pylab_examples/fill_between_demo.py
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

ax1.fill_between(x, 0, y1)
ax1.set_ylabel('between y1 and 0')

ax2.fill_between(x, y1, 1)
ax2.set_ylabel('between y1 and 1')

ax3.fill_between(x, y1, y2)
ax3.set_ylabel('between y1 and y2')
ax3.set_xlabel('x')

# now fill between y1 and y2 where a logical condition is met.  Note
# this is different than calling
#   fill_between(x[where], y1[where],y2[where]
# because of edge effects over multiple contiguous regions.
fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)
ax.plot(x, y1, x, y2, color='black')
ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='green', interpolate=True)
ax.fill_between(x, y1, y2, where=y2<=y1, facecolor='red', interpolate=True)
ax.set_title('fill between where')

# Test support for masked arrays.
y2 = np.ma.masked_greater(y2, 1.0)
ax1.plot(x, y1, x, y2, color='black')
ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='green', interpolate=True)
ax1.fill_between(x, y1, y2, where=y2<=y1, facecolor='red', interpolate=True)
ax1.set_title('Now regions with y2>1 are masked')

# This example illustrates a problem; because of the data
# gridding, there are undesired unfilled triangles at the crossover
# points.  A brute-force solution would be to interpolate all
# arrays to a very fine grid before plotting.

# show how to use transforms to create axes spans where a certain condition is satisfied
fig, ax = plt.subplots()
y = np.sin(4*np.pi*x)
ax.plot(x, y, color='black')

# use the data coordinates for the x-axis and the axes coordinates for the y-axis
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.9
ax.axhline(theta, color='green', lw=2, alpha=0.5)
ax.axhline(-theta, color='red', lw=2, alpha=0.5)
ax.fill_between(x, 0, 1, where=y>theta, facecolor='green', alpha=0.5, transform=trans)
ax.fill_between(x, 0, 1, where=y<-theta, facecolor='red', alpha=0.5, transform=trans)


plt.show()




********************************************
./pylab_examples/major_minor_demo1.py
#!/usr/bin/env python
"""
Demonstrate how to use major and minor tickers.

The two relevant userland classes are Locators and Formatters.
Locators determine where the ticks are and formatters control the
formatting of ticks.

Minor ticks are off by default (NullLocator and NullFormatter).  You
can turn minor ticks on w/o labels by setting the minor locator.  You
can also turn labeling on for the minor ticker by setting the minor
formatter

Make a plot with major ticks that are multiples of 20 and minor ticks
that are multiples of 5.  Label major ticks with %d formatting but
don't label minor ticks

The MultipleLocator ticker class is used to place ticks on multiples of
some base.  The FormatStrFormatter uses a string format string (eg
'%d' or '%1.2f' or '%1.1f cm' ) to format the tick

The pylab interface grid command changes the grid settings of the
major ticks of the y and y axis together.  If you want to control the
grid of the minor ticks for a given axis, use for example

  ax.xaxis.grid(True, which='minor')

Note, you should not use the same locator between different Axis
because the locator stores references to the Axis data and view limits

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorLocator   = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(5)


t = np.arange(0.0, 100.0, 0.1)
s = np.sin(0.1*np.pi*t)*np.exp(-t*0.01)

fig, ax = plt.subplots()
plt.plot(t,s)

ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

#for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)

plt.show()




********************************************
./pylab_examples/ginput_manual_clabel.py
#!/usr/bin/env python
# -*- noplot -*-

from __future__ import print_function
"""
This provides examples of uses of interactive functions, such as ginput,
waitforbuttonpress and manual clabel placement.

This script must be run interactively using a backend that has a
graphical user interface (for example, using GTKAgg backend, but not
PS backend).

See also ginput_demo.py
"""
import time
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def tellme(s):
    print(s)
    plt.title(s,fontsize=16)
    plt.draw()

##################################################
# Define a triangle by clicking three points
##################################################
plt.clf()
plt.axis([-1.,1.,-1.,1.])
plt.setp(plt.gca(),autoscale_on=False)

tellme('You will define a triangle, click to begin')

plt.waitforbuttonpress()

happy = False
while not happy:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')
        pts = np.asarray( plt.ginput(3,timeout=-1) )
        if len(pts) < 3:
            tellme('Too few points, starting over')
            time.sleep(1) # Wait a second

    ph = plt.fill( pts[:,0], pts[:,1], 'r', lw=2 )

    tellme('Happy? Key click for yes, mouse click for no')

    happy = plt.waitforbuttonpress()

    # Get rid of fill
    if not happy:
        for p in ph: p.remove()

##################################################
# Now contour according to distance from triangle
# corners - just an example
##################################################

# Define a nice function of distance from individual pts
def f(x,y,pts):
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1/(np.sqrt((x-p[0])**2+(y-p[1])**2))
    return 1/z

X,Y = np.meshgrid( np.linspace(-1,1,51), np.linspace(-1,1,51) )
Z = f(X,Y,pts)

CS = plt.contour( X, Y, Z, 20 )

tellme( 'Use mouse to select contour label locations, middle button to finish' )
CL = plt.clabel( CS, manual=True )

##################################################
# Now do a zoom
##################################################
tellme( 'Now do a nested zoom, click to begin' )
plt.waitforbuttonpress()

happy = False
while not happy:
    tellme( 'Select two corners of zoom, middle mouse button to finish' )
    pts = np.asarray( plt.ginput(2,timeout=-1) )

    happy = len(pts) < 2
    if happy: break

    pts = np.sort(pts,axis=0)
    plt.axis( pts.T.ravel() )

tellme('All Done!')
plt.show()




********************************************
./pylab_examples/simple_plot_fps.py
#!/usr/bin/env python

"""
Example: simple line plot.
Show how to make and save a simple line plot with labels, title and grid
"""
# -*- noplot -*-
from __future__ import print_function
from pylab import *

ion()

t = arange(0.0, 1.0+0.001, 0.001)
s = cos(2*2*pi*t)
plot(t, s, '-', lw=2)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)

import time

frames = 100.0
t = time.time()
c = time.clock()
for i in range(int(frames)):
    part = i / frames
    axis([0.0, 1.0 - part, -1.0 + part, 1.0 - part])
wallclock = time.time() - t
user = time.clock() - c
print ("wallclock:", wallclock)
print ("user:", user)
print ("fps:", frames / wallclock)




********************************************
./pylab_examples/annotation_demo.py
"""
Some examples of how to annotate points in figures.  You specify an
annotation point xy=(x,y) and a text point xytext=(x,y) for the
annotated points and text location, respectively.  Optionally, you can
specify the coordinate system of xy and xytext with one of the
following strings for xycoords and textcoords (default is 'data')


  'figure points'   : points from the lower left corner of the figure
  'figure pixels'   : pixels from the lower left corner of the figure
  'figure fraction' : 0,0 is lower left of figure and 1,1 is upper, right
  'axes points'     : points from lower left corner of axes
  'axes pixels'     : pixels from lower left corner of axes
  'axes fraction'   : 0,1 is lower left of axes and 1,1 is upper right
  'offset points'   : Specify an offset (in points) from the xy value
  'data'            : use the axes data coordinate system

Optionally, you can specify arrow properties which draws and arrow
from the text to the annotated point by giving a dictionary of arrow
properties

Valid keys are

          width : the width of the arrow in points
          frac  : the fraction of the arrow length occupied by the head
          headwidth : the width of the base of the arrow head in points
          shrink : move the tip and base some percent away from the
                   annotated point and text
          any key for matplotlib.patches.polygon  (eg facecolor)

For physical coordinate systems (points or pixels) the origin is the
(bottom, left) of the figure or axes.  If the value is negative,
however, the origin is from the (right, top) of the figure or axes,
analogous to negative indexing of sequences.
"""


from matplotlib.pyplot import figure, show
from matplotlib.patches import Ellipse
import numpy as np


if 1:
    # if only one location is given, the text and xypoint being
    # annotated are assumed to be the same
    fig = figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5))

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2*np.pi*t)
    line, = ax.plot(t, s, lw=3, color='purple')

    ax.annotate('axes center', xy=(.5, .5),  xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')

    ax.annotate('pixels', xy=(20, 20),  xycoords='figure pixels')

    ax.annotate('points', xy=(100, 300),  xycoords='figure points')

    ax.annotate('offset', xy=(1, 1),  xycoords='data',
                xytext=(-15, 10), textcoords='offset points',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='bottom',
                )

    ax.annotate('local max', xy=(3, 1),  xycoords='data',
                xytext=(0.8, 0.95), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    ax.annotate('a fractional title', xy=(.025, .975),
                xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=20)

    # use negative points or pixels to specify from right, top -10, 10
    # is 10 points to the left of the right side of the axes and 10
    # points above the bottom
    ax.annotate('bottom right (points)', xy=(-10, 10),
                xycoords='axes points',
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=20)


if 1:
    # you can specify the xypoint and the xytext in different
    # positions and coordinate systems, and optionally turn on a
    # connecting line and mark the point with a marker.  Annotations
    # work on polar axes too.  In the example below, the xy point is
    # in native coordinates (xycoords defaults to 'data').  For a
    # polar axes, this is in (theta, radius) space.  The text in this
    # example is placed in the fractional figure coordinate system.
    # Text keyword args like horizontal and vertical alignment are
    # respected
    fig = figure()
    ax = fig.add_subplot(111, polar=True)
    r = np.arange(0,1,0.001)
    theta = 2*2*np.pi*r
    line, = ax.plot(theta, r, color='#ee8d18', lw=3)

    ind = 800
    thisr, thistheta = r[ind], theta[ind]
    ax.plot([thistheta], [thisr], 'o')
    ax.annotate('a polar annotation',
                xy=(thistheta, thisr),  # theta, radius
                xytext=(0.05, 0.05),    # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom',
                )


if 1:
    # You can also use polar notation on a cartesian axes.  Here the
    # native coordinate system ('data') is cartesian, so you need to
    # specify the xycoords and textcoords as 'polar' if you want to
    # use (theta, radius)

    el = Ellipse((0,0), 10, 20, facecolor='r', alpha=0.5)

    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(el)
    el.set_clip_box(ax.bbox)
    ax.annotate('the top',
                xy=(np.pi/2., 10.),      # theta, radius
                xytext=(np.pi/3, 20.),   # theta, radius
                xycoords='polar',
                textcoords='polar',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom',
                clip_on=True, # clip to the axes bounding box
     )

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

show()




********************************************
./pylab_examples/pythonic_matplotlib.py
#!/usr/bin/env python
"""
Some people prefer to write more pythonic, object oriented, code
rather than use the pylab interface to matplotlib.  This example shows
you how.

Unless you are an application developer, I recommend using part of the
pylab interface, particularly the figure, close, subplot, axes, and
show commands.  These hide a lot of complexity from you that you don't
need to see in normal figure creation, like instantiating DPI
instances, managing the bounding boxes of the figure elements,
creating and reaslizing GUI windows and embedding figures in them.


If you are an application developer and want to embed matplotlib in
your application, follow the lead of examples/embedding_in_wx.py,
examples/embedding_in_gtk.py or examples/embedding_in_tk.py.  In this
case you will want to control the creation of all your figures,
embedding them in application windows, etc.

If you are a web application developer, you may want to use the
example in webapp_demo.py, which shows how to use the backend agg
figure canvase directly, with none of the globals (current figure,
current axes) that are present in the pylab interface.  Note that
there is no reason why the pylab interface won't work for web
application developers, however.

If you see an example in the examples dir written in pylab interface,
and you want to emulate that using the true python method calls, there
is an easy mapping.  Many of those examples use 'set' to control
figure properties.  Here's how to map those commands onto instance
methods

The syntax of set is

  setp(object or sequence, somestring, attribute)

if called with an object, set calls

  object.set_somestring(attribute)

if called with a sequence, set does

  for object in sequence:
       object.set_somestring(attribute)

So for your example, if a is your axes object, you can do

  a.set_xticklabels([])
  a.set_yticklabels([])
  a.set_xticks([])
  a.set_yticks([])
"""


from pylab import figure, show
from numpy import arange, sin, pi

t = arange(0.0, 1.0, 0.01)

fig = figure(1)

ax1 = fig.add_subplot(211)
ax1.plot(t, sin(2*pi*t))
ax1.grid(True)
ax1.set_ylim( (-2,2) )
ax1.set_ylabel('1 Hz')
ax1.set_title('A sine wave or two')

for label in ax1.get_xticklabels():
    label.set_color('r')


ax2 = fig.add_subplot(212)
ax2.plot(t, sin(2*2*pi*t))
ax2.grid(True)
ax2.set_ylim( (-2,2) )
l = ax2.set_xlabel('Hi mom')
l.set_color('g')
l.set_fontsize('large')

show()




********************************************
./pylab_examples/axes_props.py
#!/usr/bin/env python
"""
You can control the axis tick and grid properties
"""

from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s)
grid(True)

# MATLAB style
xticklines = getp(gca(), 'xticklines')
yticklines = getp(gca(), 'yticklines')
xgridlines = getp(gca(), 'xgridlines')
ygridlines = getp(gca(), 'ygridlines')
xticklabels = getp(gca(), 'xticklabels')
yticklabels = getp(gca(), 'yticklabels')

setp(xticklines, 'linewidth', 3)
setp(yticklines, 'linewidth', 3)
setp(xgridlines, 'linestyle', '-')
setp(ygridlines, 'linestyle', '-')
setp(yticklabels, 'color', 'r', fontsize='medium')
setp(xticklabels, 'color', 'r', fontsize='medium')



show()


"""
# the same script, python style
from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
fig, ax = plt.subplots()
ax.plot(t, s)
ax.grid(True)

ticklines = ax.get_xticklines()
ticklines.extend( ax.get_yticklines() )
gridlines = ax.get_xgridlines()
gridlines.extend( ax.get_ygridlines() )
ticklabels = ax.get_xticklabels()
ticklabels.extend( ax.get_yticklabels() )

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('r')
    label.set_fontsize('medium')

show()

"""




********************************************
./pylab_examples/legend_demo.py
"""
Demo of the legend function with a few features.

In addition to the basic legend, this demo shows a few optional features:

    * Custom legend placement.
    * A keyword argument to a drop-shadow.
    * Setting the background color.
    * Setting the font size.
    * Setting the line width.
"""
import numpy as np
import matplotlib.pyplot as plt


# Example data
a = np.arange(0,3, .02)
b = np.arange(0,3, .02)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c+d, 'k', label='Total message length')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()




********************************************
./pylab_examples/ellipse_collection.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

x = np.arange(10)
y = np.arange(15)
X, Y = np.meshgrid(x, y)

XY = np.hstack((X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis]))

ww = X/10.0
hh = Y/15.0
aa = X*9


fig, ax = plt.subplots()

ec = EllipseCollection(
                        ww,
                        hh,
                        aa,
                        units='x',
                        offsets=XY,
                        transOffset=ax.transData)
ec.set_array((X+Y).ravel())
ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('X')
ax.set_ylabel('y')
cbar = plt.colorbar(ec)
cbar.set_label('X+Y')
plt.show()






********************************************
./pylab_examples/quadmesh_demo.py
#!/usr/bin/env python
"""
pcolormesh uses a QuadMesh, a faster generalization of pcolor, but
with some restrictions.

This demo illustrates a bug in quadmesh with masked data.
"""

import numpy as np
from matplotlib.pyplot import figure, show, savefig
from matplotlib import cm, colors
from numpy import ma

n = 12
x = np.linspace(-1.5,1.5,n)
y = np.linspace(-1.5,1.5,n*2)
X,Y = np.meshgrid(x,y);
Qx = np.cos(Y) - np.cos(X)
Qz = np.sin(Y) + np.sin(X)
Qx = (Qx + 1.1)
Z = np.sqrt(X**2 + Y**2)/5;
Z = (Z - Z.min()) / (Z.max() - Z.min())

# The color array can include masked values:
Zm = ma.masked_where(np.fabs(Qz) < 0.5*np.amax(Qz), Z)


fig = figure()
ax = fig.add_subplot(121)
ax.set_axis_bgcolor("#bdb76b")
ax.pcolormesh(Qx,Qz,Z, shading='gouraud')
ax.set_title('Without masked values')

ax = fig.add_subplot(122)
ax.set_axis_bgcolor("#bdb76b")
#  You can control the color of the masked region:
#cmap = cm.jet
#cmap.set_bad('r', 1.0)
#ax.pcolormesh(Qx,Qz,Zm, cmap=cmap)
#  Or use the default, which is transparent:
col = ax.pcolormesh(Qx,Qz,Zm,shading='gouraud')
ax.set_title('With masked values')


show()




********************************************
./pylab_examples/log_bar.py
#!/usr/bin/env python

from matplotlib import pylab

data = ((3,1000), (10,3), (100,30), (500, 800), (50,1))

pylab.xlabel("FOO")
pylab.ylabel("FOO")
pylab.title("Testing")
pylab.gca().set_yscale('log')

dim = len(data[0])
w = 0.75
dimw = w / dim

x = pylab.arange(len(data))
for i in range(len(data[0])) :
    y = [d[i] for d in data]
    b = pylab.bar(x + i * dimw, y, dimw, bottom=0.001)
pylab.gca().set_xticks(x + w / 2)
pylab.gca().set_ylim( (0.001,1000))

pylab.show()






********************************************
./pylab_examples/custom_ticker1.py
#!/usr/bin/env python

"""
The new ticker code was designed to explicitly support user customized
ticking.  The documentation
http://matplotlib.org/matplotlib.ticker.html details this
process.  That code defines a lot of preset tickers but was primarily
designed to be user extensible.

In this example a user defined function is used to format the ticks in
millions of dollars on the y axis
"""
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x =     np.arange(4)
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x*1e-6)

formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(x, money)
plt.xticks( x + 0.5,  ('Bill', 'Fred', 'Mary', 'Sue') )
plt.show()




********************************************
./pylab_examples/tricontour_vs_griddata.py
"""
Comparison of griddata and tricontour for an unstructured triangular grid.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import time

seed(0)
npts = 200
ngridx = 100
ngridy = 200
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)

# griddata and contour.
start = time.clock()
plt.subplot(211)
xi = np.linspace(-2.1, 2.1, ngridx)
yi = np.linspace(-2.1, 2.1, ngridy)
zi = griddata(x, y, z, xi, yi, interp='linear')
plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
             norm=plt.Normalize(vmax=abs(zi).max(), vmin=-abs(zi).max()))
plt.colorbar()  # draw colorbar
plt.plot(x, y, 'ko', ms=3)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('griddata and contour (%d points, %d grid points)' %
          (npts, ngridx*ngridy))
print ('griddata and contour seconds: %f' % (time.clock() - start))

# tricontour.
start = time.clock()
plt.subplot(212)
triang = tri.Triangulation(x, y)
plt.tricontour(x, y, z, 15, linewidths=0.5, colors='k')
plt.tricontourf(x, y, z, 15, cmap=plt.cm.rainbow,
                norm=plt.Normalize(vmax=abs(zi).max(), vmin=-abs(zi).max()))
plt.colorbar()
plt.plot(x, y, 'ko', ms=3)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('tricontour (%d points)' % npts)
print ('tricontour seconds: %f' % (time.clock() - start))

plt.show()




********************************************
./pylab_examples/spy_demos.py
"""
Plot the sparsity pattern of arrays
"""

from matplotlib.pyplot import figure, show
import numpy

fig = figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

x = numpy.random.randn(20,20)
x[5] = 0.
x[:,12] = 0.

ax1.spy(x, markersize=5)
ax2.spy(x, precision=0.1, markersize=5)

ax3.spy(x)
ax4.spy(x, precision=0.1)

show()




********************************************
./pylab_examples/scatter_demo2.py
"""
Demo of scatter plot with varying marker colors and sizes.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

# Load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column
datafile = cbook.get_sample_data('goog.npy')
price_data = np.load(datafile).view(np.recarray)
price_data = price_data[-250:] # get the most recent 250 trading days

delta1 = np.diff(price_data.adj_close)/price_data.adj_close[:-1]

# Marker size in units of points^2
volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=20)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=20)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()




********************************************
./pylab_examples/dannys_example.py
import matplotlib
matplotlib.rc('text', usetex = True)
import pylab
import numpy as np

## interface tracking profiles
N = 500
delta = 0.6
X = -1 + 2. * np.arange(N) / (N - 1)
pylab.plot(X, (1 - np.tanh(4. * X / delta)) / 2,     ## phase field tanh profiles
           X, (X + 1) / 2,                           ## level set distance function
           X, (1.4 + np.tanh(4. * X / delta)) / 4,   ## composition profile
           X, X < 0, 'k--',                               ## sharp interface
           linewidth = 5)

## legend
pylab.legend((r'phase field', r'level set', r'composition', r'sharp interface'), shadow = True, loc = (0.01, 0.55))
ltext = pylab.gca().get_legend().get_texts()
pylab.setp(ltext[0], fontsize = 20, color = 'b')
pylab.setp(ltext[1], fontsize = 20, color = 'g')
pylab.setp(ltext[2], fontsize = 20, color = 'r')
pylab.setp(ltext[3], fontsize = 20, color = 'k')

## the arrow
height = 0.1
offset = 0.02
pylab.plot((-delta / 2., delta / 2), (height, height), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.text(-0.06, height - 0.06, r'$\delta$', {'color' : 'k', 'fontsize' : 24})

## X-axis label
pylab.xticks((-1, 0,  1), ('-1', '0',  '1'), color = 'k', size = 20)

## Left Y-axis labels
pylab.ylabel(r'\bf{phase field} $\phi$', {'color'    : 'b',
                                          'fontsize'   : 20 })
pylab.yticks((0, 0.5, 1), ('0', '.5', '1'), color = 'k', size = 20)

## Right Y-axis labels
pylab.text(1.05, 0.5, r"\bf{level set} $\phi$", {'color' : 'g', 'fontsize' : 20},
           horizontalalignment = 'left',
           verticalalignment = 'center',
           rotation = 90,
           clip_on = False)
pylab.text(1.01, -0.02, "-1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.98, "1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.48, "0", {'color' : 'k', 'fontsize' : 20})

## level set equations
pylab.text(0.1, 0.85, r'$|\nabla\phi| = 1,$ \newline $ \frac{\partial \phi}{\partial t} + U|\nabla \phi| = 0$', {'color' : 'g', 'fontsize' : 20})

## phase field equations
pylab.text(0.2, 0.15, r'$\mathcal{F} = \int f\left( \phi, c \right) dV,$ \newline $ \frac{ \partial \phi } { \partial t } = -M_{ \phi } \frac{ \delta \mathcal{F} } { \delta \phi }$',
           {'color' : 'b', 'fontsize' : 20})

pylab.show()




********************************************
./pylab_examples/scatter_custom_symbol.py
import matplotlib.pyplot as plt
from numpy import arange, pi, cos, sin
from numpy.random import rand

# unit area ellipse
rx, ry = 3., 1.
area = rx * ry * pi
theta = arange(0, 2*pi+0.01, 0.1)
verts = list(zip(rx/area*cos(theta), ry/area*sin(theta)))

x,y,s,c = rand(4, 30)
s*= 10**2.

fig, ax = plt.subplots()
ax.scatter(x,y,s,c,marker=None,verts =verts)

plt.show()




********************************************
./pylab_examples/specgram_demo.py
#!/usr/bin/env python
from pylab import *

dt = 0.0005
t = arange(0.0, 20.0, dt)
s1 = sin(2*pi*100*t)
s2 = 2*sin(2*pi*400*t)

# create a transient "chirp"
mask = where(logical_and(t>10, t<12), 1.0, 0.0)
s2 = s2 * mask

# add some noise into the mix
nse = 0.01*randn(len(t))

x = s1 + s2 + nse # the signal
NFFT = 1024       # the length of the windowing segments
Fs = int(1.0/dt)  # the sampling frequency

# Pxx is the segments x freqs array of instantaneous power, freqs is
# the frequency vector, bins are the centers of the time bins in which
# the power is computed, and im is the matplotlib.image.AxesImage
# instance

ax1 = subplot(211)
plot(t, x)
subplot(212, sharex=ax1)           
Pxx, freqs, bins, im = specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900, 
                                cmap=cm.gist_heat)
show()




********************************************
./pylab_examples/broken_barh.py
"""
Make a "broken" horizontal bar plot, ie one with gaps
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.broken_barh([ (110, 30), (150, 10) ] , (10, 9), facecolors='blue')
ax.broken_barh([ (10, 50), (100, 20),  (130, 10)] , (20, 9),
                facecolors=('red', 'yellow', 'green'))
ax.set_ylim(5,35)
ax.set_xlim(0,200)
ax.set_xlabel('seconds since start')
ax.set_yticks([15,25])
ax.set_yticklabels(['Bill', 'Jim'])
ax.grid(True)
ax.annotate('race interrupted', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')

plt.show()




********************************************
./pylab_examples/major_minor_demo2.py
#!/usr/bin/env python
"""
Automatic tick selection for major and minor ticks.

Use interactive pan and zoom to see how the tick intervals
change. There will be either 4 or 5 minor tick intervals
per major interval, depending on the major interval.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# One can supply an argument to AutoMinorLocator to
# specify a fixed number of minor intervals per major interval, e.g.:
# minorLocator = AutoMinorLocator(2)
# would lead to a single minor tick between major ticks.

minorLocator   = AutoMinorLocator()


t = np.arange(0.0, 100.0, 0.01)
s = np.sin(2*np.pi*t)*np.exp(-t*0.01)

fig, ax = plt.subplots()
plt.plot(t,s)

ax.xaxis.set_minor_locator(minorLocator)

plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='r')

plt.show()




********************************************
./pylab_examples/pstest.py
#!/usr/bin/env python
# -*- noplot -*-
import matplotlib
matplotlib.use('PS')
from pylab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .1)
t2 = arange(0.0, 5.0, 0.02)
t3 = arange(0.0, 2.0, 0.01)

figure(1)
subplot(211)
l = plot(t1, f(t1), 'k^')
setp(l, markerfacecolor='k', markeredgecolor='r')
title('A tale of 2 subplots', fontsize=14, fontname='Courier')
ylabel('Signal 1', fontsize=12)
subplot(212)
l = plot(t1, f(t1), 'k>')


ylabel('Signal 2', fontsize=12)
xlabel('time (s)', fontsize=12)

show()





********************************************
./pylab_examples/multi_image.py
#!/usr/bin/env python
'''
Make a set of images with a single colormap, norm, and colorbar.

It also illustrates colorbar tick labelling with a multiplier.
'''

from matplotlib.pyplot import figure, show, axes, sci
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from numpy import amin, amax, ravel
from numpy.random import rand

Nr = 3
Nc = 2

fig = figure()
cmap = cm.cool

figtitle = 'Multiple images'
t = fig.text(0.5, 0.95, figtitle,
               horizontalalignment='center',
               fontproperties=FontProperties(size=16))

cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])

w = 0.4
h = 0.22
ax = []
images = []
vmin = 1e40
vmax = -1e40
for i in range(Nr):
    for j in range(Nc):
        pos = [0.075 + j*1.1*w, 0.18 + i*1.2*h, w, h]
        a = fig.add_axes(pos)
        if i > 0:
            a.set_xticklabels([])
        # Make some fake data with a range that varies
        # somewhat from one plot to the next.
        data =((1+i+j)/10.0)*rand(10,20)*1e-6
        dd = ravel(data)
        # Manually find the min and max of all colors for
        # use in setting the color scale.
        vmin = min(vmin, amin(dd))
        vmax = max(vmax, amax(dd))
        images.append(a.imshow(data, cmap=cmap))

        ax.append(a)

# Set the first image as the master, with all the others
# observing it for changes in cmap or norm.

class ImageFollower:
    'update image in response to changes in clim or cmap on another image'
    def __init__(self, follower):
        self.follower = follower
    def __call__(self, leader):
        self.follower.set_cmap(leader.get_cmap())
        self.follower.set_clim(leader.get_clim())

norm = colors.Normalize(vmin=vmin, vmax=vmax)
for i, im in enumerate(images):
    im.set_norm(norm)
    if i > 0:
        images[0].callbacksSM.connect('changed', ImageFollower(im))

# The colorbar is also based on this master image.
fig.colorbar(images[0], cax, orientation='horizontal')

# We need the following only if we want to run this interactively and
# modify the colormap:

axes(ax[0])     # Return the current axes to the first one,
sci(images[0])  # because the current image must be in current axes.

show()










********************************************
./pylab_examples/cursor_demo.py
#!/usr/bin/env python
# -*- noplot -*-

"""

This example shows how to use matplotlib to provide a data cursor.  It
uses matplotlib to draw the cursor and may be a slow since this
requires redrawing the figure with every mouse move.

Faster cursoring is possible using native GUI drawing, as in
wxcursor_demo.py
"""
from __future__ import print_function
from pylab import *


class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        draw()


class SnaptoCursor:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes: return

        x, y = event.xdata, event.ydata

        indx = searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        print ('x=%1.2f, y=%1.2f'%(x,y))
        draw()

t = arange(0.0, 1.0, 0.01)
s = sin(2*2*pi*t)
fig, ax = subplots()

cursor = Cursor(ax)
#cursor = SnaptoCursor(ax, t, s)
connect('motion_notify_event', cursor.mouse_move)

ax.plot(t, s, 'o')
axis([0,1,-1,1])
show()





********************************************
./pylab_examples/shared_axis_demo.py
"""
You can share the x or y axis limits for one axis with another by
passing an axes instance as a sharex or sharey kwarg.

Changing the axis limits on one axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the axes will follow each other on their shared axes.  Ditto for
changes in the axis scaling (eg log vs linear).  However, it is
possible to have differences in tick labeling, eg you can selectively
turn off the tick labels on one axes.

The example below shows how to customize the tick labels on the
various axes.  Shared axes share the tick locator, tick formatter,
view limits, and transformation (eg log, linear).  But the ticklabels
themselves do not share properties.  This is a feature and not a bug,
because you may want to make the tick labels smaller on the upper
axes, eg in the example below.

If you want to turn off the ticklabels for a given axes (eg on
subplot(211) or subplot(212), you cannot do the standard trick

   setp(ax2, xticklabels=[])

because this changes the tick Formatter, which is shared among all
axes.  But you can alter the visibility of the labels, which is a
property

  setp( ax2.get_xticklabels(), visible=False)


"""
from pylab import *

t = arange(0.01, 5.0, 0.01)
s1 = sin(2*pi*t)
s2 = exp(-t)
s3 = sin(4*pi*t)
ax1 = subplot(311)
plot(t,s1)
setp( ax1.get_xticklabels(), fontsize=6)

## share x only
ax2 = subplot(312, sharex=ax1)
plot(t, s2)
# make these tick labels invisible
setp( ax2.get_xticklabels(), visible=False)

# share x and y
ax3 = subplot(313,  sharex=ax1, sharey=ax1)
plot(t, s3)
xlim(0.01,5.0)
show()




********************************************
./pylab_examples/system_monitor.py
#!/usr/bin/env python
# -*- noplot -*-
import time
from pylab import *

def get_memory():
    "Simulate a function that returns system memory"
    return 100*(0.5+0.5*sin(0.5*pi*time.time()))

def get_cpu():
    "Simulate a function that returns cpu usage"
    return 100*(0.5+0.5*sin(0.2*pi*(time.time()-0.25)))

def get_net():
    "Simulate a function that returns network bandwidth"
    return 100*(0.5+0.5*sin(0.7*pi*(time.time()-0.1)))

def get_stats():
    return get_memory(), get_cpu(), get_net()

# turn interactive mode on for dynamic updates.  If you aren't in
# interactive mode, you'll need to use a GUI event handler/timer.
ion()

fig, ax = plt.subplots()
ind = arange(1,4)
pm, pc, pn = bar(ind, get_stats())
centers = ind + 0.5*pm.get_width()
pm.set_facecolor('r')
pc.set_facecolor('g')
pn.set_facecolor('b')
ax.set_xlim([0.5,4])
ax.set_xticks(centers)
ax.set_ylim([0,100])
ax.set_xticklabels(['Memory', 'CPU', 'Bandwidth'])
ax.set_ylabel('Percent usage')
ax.set_title('System Monitor')

for i in range(200):  # run for a little while
    m,c,n = get_stats()

    pm.set_height(m)
    pc.set_height(c)
    pn.set_height(n)
    ax.set_ylim([0,100])

    draw()









********************************************
./pylab_examples/fill_spiral.py
#!/usr/bin/env python

from pylab import *

theta = arange(0,8*pi,0.1)
a=1
b=.2

for dt in arange(0,2*pi,pi/2.0):

     x = a*cos( theta+dt )*exp(b*theta)
     y = a*sin( theta+dt )*exp(b*theta)

     dt = dt+pi/4.0

     x2 = a*cos( theta+dt )*exp(b*theta)
     y2 = a*sin( theta+dt )*exp(b*theta)

     xf = concatenate( (x,x2[::-1]) )
     yf = concatenate( (y,y2[::-1]) )

     p1=fill(xf,yf)

show()




********************************************
./pylab_examples/coords_report.py
#!/usr/bin/env python

# override the default reporting of coords

from pylab import *

def millions(x):
    return '$%1.1fM' % (x*1e-6)

x =     rand(20)
y =     1e7*rand(20)

fig, ax = subplots()
ax.fmt_ydata = millions
plot(x, y, 'o')

show()





********************************************
./axes_grid/demo_parasite_axes2.py
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

if 1:

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))
        
    par2.axis["right"].toggle(all=True)



    host.set_xlim(0, 2)
    host.set_ylim(0, 2)

    host.set_xlabel("Distance")
    host.set_ylabel("Density")
    par1.set_ylabel("Temperature")
    par2.set_ylabel("Velocity")

    p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
    p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")
    p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    plt.draw()
    plt.show()

    #plt.savefig("Test")




********************************************
./axes_grid/simple_axisline4.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import numpy as np

ax = host_subplot(111, axes_class=AA.Axes)
xx = np.arange(0, 2*np.pi, 0.01)
ax.plot(xx, np.sin(xx))

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
                     r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

ax2.axis["right"].major_ticklabels.set_visible(False)

plt.draw()
plt.show()





********************************************
./axes_grid/demo_floating_axes.py
from matplotlib.transforms import Affine2D

import mpl_toolkits.axisartist.floating_axes as floating_axes

import numpy as np
import  mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, \
     DictFormatter

def setup_axes1(fig, rect):
    """
    A simple one.
    """
    tr = Affine2D().scale(2, 1).rotate_deg(30)

    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(0, 4, 0, 4))

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    aux_ax = ax1.get_aux_axes(tr)

    grid_helper.grid_finder.grid_locator1._nbins = 4
    grid_helper.grid_finder.grid_locator2._nbins = 4

    return ax1, aux_ax


def setup_axes2(fig, rect):
    """
    With custom locator and formatter.
    Note that the extreme values are swapped.
    """

    #tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = PolarAxes.PolarTransform()

    pi = np.pi
    angle_ticks = [(0, r"$0$"),
                   (.25*pi, r"$\frac{1}{4}\pi$"),
                   (.5*pi, r"$\frac{1}{2}\pi$")]
    grid_locator1 = FixedLocator([v for v, s in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))

    grid_locator2 = MaxNLocator(2)

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                        extremes=(.5*pi, 0, 2, 1),
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.

    return ax1, aux_ax


def setup_axes3(fig, rect):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(-95, 0)

    # scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorHMS(4)
    tick_formatter1 = angle_helper.FormatterHMS()

    grid_locator2 = MaxNLocator(3)

    ra0, ra1 = 8.*15, 14.*15
    cz0, cz1 = 0, 14000
    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                        extremes=(ra0, ra1, cz0, cz1),
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    ax1.axis["left"].label.set_text(r"cz [km$^{-1}$]")
    ax1.axis["top"].label.set_text(r"$\alpha_{1950}$")


    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.

    return ax1, aux_ax



if 1:
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(8, 4))
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    ax1, aux_ax2 = setup_axes1(fig, 131)
    aux_ax2.bar([0, 1, 2, 3], [3, 2, 1, 3])
    
    #theta = np.random.rand(10) #*.5*np.pi
    #radius = np.random.rand(10) #+1.
    #aux_ax1.scatter(theta, radius)


    ax2, aux_ax2 = setup_axes2(fig, 132)

    theta = np.random.rand(10)*.5*np.pi
    radius = np.random.rand(10)+1.
    aux_ax2.scatter(theta, radius)


    ax3, aux_ax3 = setup_axes3(fig, 133)

    theta = (8 + np.random.rand(10)*(14-8))*15. # in degrees
    radius = np.random.rand(10)*14000.
    aux_ax3.scatter(theta, radius)

    plt.show()





********************************************
./axes_grid/inset_locator_demo.py
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def add_sizebar(ax, size):
   asb =  AnchoredSizeBar(ax.transData,
                         size,
                         str(size),
                         loc=8,
                         pad=0.1, borderpad=0.5, sep=5,
                         frameon=False)
   ax.add_artist(asb)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=[5.5, 3])

# first subplot
ax.set_aspect(1.)

axins = inset_axes(ax,
                   width="30%", # width = 30% of parent_bbox
                   height=1., # height : 1 inch
                   loc=3)

plt.xticks(visible=False)
plt.yticks(visible=False)


# second subplot
ax2.set_aspect(1.)

axins = zoomed_inset_axes(ax2, 0.5, loc=1) # zoom = 0.5

plt.xticks(visible=False)
plt.yticks(visible=False)

add_sizebar(ax2, 0.5)
add_sizebar(axins, 0.5)

plt.draw()
plt.show()




********************************************
./axes_grid/simple_anchored_artists.py
import matplotlib.pyplot as plt


def draw_text(ax):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredText
    at = AnchoredText("Figure 1a",
                      loc=2, prop=dict(size=8), frameon=True,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    at2 = AnchoredText("Figure 1(b)",
                       loc=3, prop=dict(size=8), frameon=True,
                       bbox_to_anchor=(0., 1.),
                       bbox_transform=ax.transAxes
                       )
    at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at2)

def draw_circle(ax): # circle in the canvas coordinate
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
    from matplotlib.patches import Circle
    ada = AnchoredDrawingArea(20, 20, 0, 0,
                              loc=1, pad=0., frameon=False)
    p = Circle((10, 10), 10)
    ada.da.add_artist(p)
    ax.add_artist(ada)

def draw_ellipse(ax):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
    # draw an ellipse of width=0.1, height=0.15 in the data coordinate
    ae = AnchoredEllipse(ax.transData, width=0.1, height=0.15, angle=0.,
                         loc=3, pad=0.5, borderpad=0.4, frameon=True)

    ax.add_artist(ae)

def draw_sizebar(ax):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    # draw a horizontal bar with length of 0.1 in Data coordinate
    # (ax.transData) with a label underneath.
    asb =  AnchoredSizeBar(ax.transData,
                          0.1,
                          r"1$^{\prime}$",
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)


if 1:
    ax = plt.gca()
    ax.set_aspect(1.)

    draw_text(ax)
    draw_circle(ax)
    draw_ellipse(ax)
    draw_sizebar(ax)

    plt.show()






********************************************
./axes_grid/demo_edge_colorbar.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)


def demo_bottom_cbar(fig):
    """
    A grid of 2x2 images with a colorbar for each column.
    """
    grid = AxesGrid(fig, 121, # similar to subplot(132)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.10,
                    share_all=True,
                    label_mode = "1",
                    cbar_location = "bottom",
                    cbar_mode="edge",
                    cbar_pad = 0.25,
                    cbar_size = "15%",
                    direction="column"
                    )

    Z, extent = get_demo_image()
    cmaps = [plt.get_cmap("autumn"), plt.get_cmap("summer")]
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest",
                cmap=cmaps[i//2])
        if i % 2:
            cbar = grid.cbar_axes[i//2].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(True)
        cax.axis[cax.orientation].set_label("Bar")

    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


def demo_right_cbar(fig):
    """
    A grid of 2x2 images. Each row has its own colorbar.
    """

    grid = AxesGrid(F, 122, # similar to subplot(122)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.10,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    Z, extent = get_demo_image()
    cmaps = [plt.get_cmap("spring"), plt.get_cmap("winter")]
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest",
                cmap=cmaps[i//2])
        if i % 2:
            grid.cbar_axes[i//2].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(True)
        cax.axis[cax.orientation].set_label('Foo')

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])



if 1:
    F = plt.figure(1, (5.5, 2.5))

    F.subplots_adjust(left=0.05, right=0.93)

    demo_bottom_cbar(F)
    demo_right_cbar(F)

    plt.draw()
    plt.show()





********************************************
./axes_grid/demo_floating_axis.py
"""
An experimental support for curvilinear grid.
"""


def curvelinear_test2(fig):
    """
    polar projection, but in a rectangular box.
    """
    global ax1
    import numpy as np
    import  mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D

    from mpl_toolkits.axisartist import SubplotHost

    from mpl_toolkits.axisartist import GridHelperCurveLinear

    # see demo_curvelinear_grid.py for details
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 360,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)

    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    # Now creates floating axis

    #grid_helper = ax1.get_grid_helper()
    # floating axis whose first coordinate (theta) is fixed at 60
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    axis.label.set_text(r"$\theta = 60^{\circ}$")
    axis.label.set_visible(True)

    # floating axis whose second coordinate (r) is fixed at 6
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)

import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(5, 5))
fig.clf()

curvelinear_test2(fig)

plt.show()






********************************************
./axes_grid/parasite_simple2.py
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

obs = [["01_S1", 3.88, 0.14, 1970, 63],
       ["01_S4", 5.6, 0.82, 1622, 150],
       ["02_S1", 2.4, 0.54, 1570, 40],
       ["03_S1", 4.1, 0.62, 2380, 170]]


fig = plt.figure()

ax_kms = SubplotHost(fig, 1,1,1, aspect=1.)

# angular proper motion("/yr) to linear velocity(km/s) at distance=2.3kpc
pm_to_kms = 1./206265.*2300*3.085e18/3.15e7/1.e5

aux_trans = mtransforms.Affine2D().scale(pm_to_kms, 1.)
ax_pm = ax_kms.twin(aux_trans)
ax_pm.set_viewlim_mode("transform")

fig.add_subplot(ax_kms)

for n, ds, dse, w, we in obs:
    time = ((2007+(10. + 4/30.)/12)-1988.5)
    v = ds / time * pm_to_kms
    ve = dse / time * pm_to_kms
    ax_kms.errorbar([v], [w], xerr=[ve], yerr=[we], color="k")


ax_kms.axis["bottom"].set_label("Linear velocity at 2.3 kpc [km/s]")
ax_kms.axis["left"].set_label("FWHM [km/s]")
ax_pm.axis["top"].set_label("Proper Motion [$^{''}$/yr]")
ax_pm.axis["top"].label.set_visible(True)
ax_pm.axis["right"].major_ticklabels.set_visible(False)

ax_kms.set_xlim(950, 3700)
ax_kms.set_ylim(950, 3100)
# xlim and ylim of ax_pms will be automatically adjusted.

plt.draw()
plt.show()




********************************************
./axes_grid/make_room_for_ylabel_using_axesgrid.py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    def ex1():
        plt.figure(1)
        ax = plt.axes([0,0,1,1])
        # ax = plt.subplot(111)

        ax.set_yticks([0.5])
        ax.set_yticklabels(["very long label"])

        make_axes_area_auto_adjustable(ax)
    

    def ex2():

        plt.figure(2)
        ax1 = plt.axes([0,0,1,0.5])
        ax2 = plt.axes([0,0.5,1,0.5])

        ax1.set_yticks([0.5])
        ax1.set_yticklabels(["very long label"])
        ax1.set_ylabel("Y label")
        
        ax2.set_title("Title")

        make_axes_area_auto_adjustable(ax1, pad=0.1, use_axes=[ax1, ax2]) 
        make_axes_area_auto_adjustable(ax2, pad=0.1, use_axes=[ax1, ax2])

    def ex3():

        fig = plt.figure(3)
        ax1 = plt.axes([0,0,1,1])
        divider = make_axes_locatable(ax1)
        
        ax2 = divider.new_horizontal("100%", pad=0.3, sharey=ax1)
        ax2.tick_params(labelleft="off")
        fig.add_axes(ax2)

        divider.add_auto_adjustable_area(use_axes=[ax1], pad=0.1,
                                         adjust_dirs=["left"])
        divider.add_auto_adjustable_area(use_axes=[ax2], pad=0.1,
                                         adjust_dirs=["right"])
        divider.add_auto_adjustable_area(use_axes=[ax1, ax2], pad=0.1,
                                         adjust_dirs=["top", "bottom"])
    
        ax1.set_yticks([0.5])
        ax1.set_yticklabels(["very long label"])

        ax2.set_title("Title")
        ax2.set_xlabel("X - Label")
    
    ex1()
    ex2()
    ex3()

    plt.show()




********************************************
./axes_grid/simple_axesgrid.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

im = np.arange(100)
im.shape = 10, 10

fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (2, 2), # creates 2x2 grid of axes
                axes_pad=0.1, # pad between axes in inch.
                )

for i in range(4):
    grid[i].imshow(im) # The AxesGrid object work as a list of axes.

plt.show()




********************************************
./axes_grid/scatter_hist.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)


fig, axScatter = plt.subplots(figsize=(5.5,5.5))

# the scatter plot:
axScatter.scatter(x, y)
axScatter.set_aspect(1.)

# create new axes on the right and on the top of the current axes
# The first argument of the new_vertical(new_horizontal) method is
# the height (width) of the axes to be created in inches.
divider = make_axes_locatable(axScatter)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# make some labels invisible
plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
         visible=False)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
lim = ( int(xymax/binwidth) + 1) * binwidth

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
for tl in axHistx.get_xticklabels():
    tl.set_visible(False)
axHistx.set_yticks([0, 50, 100])

#axHisty.axis["left"].major_ticklabels.set_visible(False)
for tl in axHisty.get_yticklabels():
    tl.set_visible(False)
axHisty.set_xticks([0, 50, 100])

plt.draw()
plt.show()




********************************************
./axes_grid/simple_axesgrid2.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)

F = plt.figure(1, (5.5, 3.5))
grid = ImageGrid(F, 111, # similar to subplot(111)
                nrows_ncols = (1, 3),
                axes_pad = 0.1,
                add_all=True,
                label_mode = "L",
                )

Z, extent = get_demo_image() # demo image

im1=Z
im2=Z[:,:10]
im3=Z[:,10:]
vmin, vmax = Z.min(), Z.max()
for i, im in enumerate([im1, im2, im3]):
    ax = grid[i]
    ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

plt.draw()
plt.show()




********************************************
./axes_grid/demo_axes_hbox_divider.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size

def make_heights_equal(fig, rect, ax1, ax2, pad):
    # pad in inches
    
    h1, v1 = Size.AxesX(ax1), Size.AxesY(ax1)
    h2, v2 = Size.AxesX(ax2), Size.AxesY(ax2)

    pad_v = Size.Scaled(1)
    pad_h = Size.Fixed(pad)

    my_divider = HBoxDivider(fig, rect,
                             horizontal=[h1, pad_h, h2],
                             vertical=[v1, pad_v, v2])


    ax1.set_axes_locator(my_divider.new_locator(0))
    ax2.set_axes_locator(my_divider.new_locator(2))

    
if __name__ == "__main__":

    arr1 = np.arange(20).reshape((4,5))
    arr2 = np.arange(20).reshape((5,4))

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(arr1, interpolation="nearest")
    ax2.imshow(arr2, interpolation="nearest")

    rect = 111 # subplot param for combined axes
    make_heights_equal(fig, rect, ax1, ax2, pad=0.5) # pad in inches
    
    for ax in [ax1, ax2]:
        ax.locator_params(nbins=4)

    # annotate
    ax3 = plt.axes([0.5, 0.5, 0.001, 0.001], frameon=False)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.annotate("Location of two axes are adjusted\n"
                 "so that they have equal heights\n"
                 "while maintaining their aspect ratios", (0.5, 0.5),
                 xycoords="axes fraction", va="center", ha="center",
                 bbox=dict(boxstyle="round, pad=1", fc="w"))

    plt.show()
    




********************************************
./axes_grid/demo_axes_rgb.py
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes

def get_demo_image():
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)



def get_rgb():
    Z, extent = get_demo_image()

    Z[Z<0] = 0.
    Z = Z/Z.max()

    R = Z[:13,:13]
    G = Z[2:,2:]
    B = Z[:13,2:]

    return R, G, B


def make_cube(r, g, b):
    ny, nx = r.shape
    R = np.zeros([ny, nx, 3], dtype="d")
    R[:,:,0] = r
    G = np.zeros_like(R)
    G[:,:,1] = g
    B = np.zeros_like(R)
    B[:,:,2] = b

    RGB = R + G + B

    return R, G, B, RGB



def demo_rgb():
    fig, ax = plt.subplots()
    ax_r, ax_g, ax_b = make_rgb_axes(ax, pad=0.02)
    #fig.add_axes(ax_r)
    #fig.add_axes(ax_g)
    #fig.add_axes(ax_b)

    r, g, b = get_rgb()
    im_r, im_g, im_b, im_rgb = make_cube(r, g, b)
    kwargs = dict(origin="lower", interpolation="nearest")
    ax.imshow(im_rgb, **kwargs)
    ax_r.imshow(im_r, **kwargs)
    ax_g.imshow(im_g, **kwargs)
    ax_b.imshow(im_b, **kwargs)




def demo_rgb2():
    fig = plt.figure(2)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
    #fig.add_axes(ax)
    #ax.add_RGB_to_figure()

    r, g, b = get_rgb()
    kwargs = dict(origin="lower", interpolation="nearest")
    ax.imshow_rgb(r, g, b, **kwargs)

    ax.RGB.set_xlim(0., 9.5)
    ax.RGB.set_ylim(0.9, 10.6)

    for ax1 in [ax.RGB, ax.R, ax.G, ax.B]:
        for sp1 in ax1.spines.values():
            sp1.set_color("w")
        for tick in ax1.xaxis.get_major_ticks() + ax1.yaxis.get_major_ticks():
            tick.tick1line.set_mec("w")
            tick.tick2line.set_mec("w")

    return ax


demo_rgb()
ax = demo_rgb2()

plt.show()




********************************************
./axes_grid/demo_curvelinear_grid.py
import numpy as np
#from matplotlib.path import Path

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot

from mpl_toolkits.axisartist import SubplotHost, \
     ParasiteAxesAuxTrans


def curvelinear_test1(fig):
    """
    grid for custom transform.
    """

    def tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return x, y-x

    def inv_tr(x,y):
        x, y = np.asarray(x), np.asarray(y)
        return x, y+x


    grid_helper = GridHelperCurveLinear((tr, inv_tr))

    ax1 = Subplot(fig, 1, 2, 1, grid_helper=grid_helper)
    # ax1 will have a ticks and gridlines defined by the given
    # transform (+ transData of the Axes). Note that the transform of
    # the Axes itself (i.e., transData) is not affected by the given
    # transform.

    fig.add_subplot(ax1)

    xx, yy = tr([3, 6], [5.0, 10.])
    ax1.plot(xx, yy)

    ax1.set_aspect(1.)
    ax1.set_xlim(0, 10.)
    ax1.set_ylim(0, 10.)

    ax1.axis["t"]=ax1.new_floating_axis(0, 3.)
    ax1.axis["t2"]=ax1.new_floating_axis(1, 7.)
    ax1.grid(True)



import  mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D

def curvelinear_test2(fig):
    """
    polar projection, but in a rectangular box.
    """

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 360,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)
    # Find a grid values appropriate for the coordinate (degree,
    # minute, second).

    tick_formatter1 = angle_helper.FormatterDMS()
    # And also uses an appropriate formatter.  Note that,the
    # acceptable Locator and Formatter class is a bit different than
    # that of mpl's, and you cannot directly use mpl's Locator and
    # Formatter here (but may be possible in the future).

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


    ax1 = SubplotHost(fig, 1, 2, 2, grid_helper=grid_helper)

    # make ticklabels of right and top axis visible.
    ax1.axis["right"].major_ticklabels.set_visible(True)
    ax1.axis["top"].major_ticklabels.set_visible(True)

    # let right axis shows ticklabels for 1st coordinate (angle)
    ax1.axis["right"].get_helper().nth_coord_ticks=0
    # let bottom axis shows ticklabels for 2nd coordinate (radius)
    ax1.axis["bottom"].get_helper().nth_coord_ticks=1

    fig.add_subplot(ax1)


    # A parasite axes with given transform
    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    # note that ax2.transData == tr + ax1.transData
    # Anthing you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)
    intp = cbook.simple_linear_interpolation
    ax2.plot(intp(np.array([0, 30]), 50),
             intp(np.array([10., 10.]), 50))

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)

if 1:
    fig = plt.figure(1, figsize=(7, 4))
    fig.clf()

    curvelinear_test1(fig)
    curvelinear_test2(fig)

    plt.draw()
    plt.show()







********************************************
./axes_grid/demo_curvelinear_grid2.py
import numpy as np
#from matplotlib.path import Path

import matplotlib.pyplot as plt

from  mpl_toolkits.axes_grid.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axes_grid.axislines import Subplot

import  mpl_toolkits.axes_grid.angle_helper as angle_helper

def curvelinear_test1(fig):
    """
    grid for custom transform.
    """

    def tr(x, y):
        sgn = np.sign(x)
        x, y = np.abs(np.asarray(x)), np.asarray(y)
        return sgn*x**.5, y

    def inv_tr(x,y):
        sgn = np.sign(x)
        x, y = np.asarray(x), np.asarray(y)
        return sgn*x**2, y

    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = None,
                                                     lat_cycle = None,
                                                     lon_minmax = None, #(0, np.inf),
                                                     lat_minmax = None,
                                                     )

    grid_helper = GridHelperCurveLinear((tr, inv_tr),
                                        extreme_finder=extreme_finder)

    ax1 = Subplot(fig, 111, grid_helper=grid_helper)
    # ax1 will have a ticks and gridlines defined by the given
    # transform (+ transData of the Axes). Note that the transform of
    # the Axes itself (i.e., transData) is not affected by the given
    # transform.

    fig.add_subplot(ax1)

    ax1.imshow(np.arange(25).reshape(5,5),
               vmax = 50, cmap=plt.cm.gray_r,
               interpolation="nearest",
               origin="lower")

    # tick density
    grid_helper.grid_finder.grid_locator1._nbins = 6
    grid_helper.grid_finder.grid_locator2._nbins = 6



if 1:
    fig = plt.figure(1, figsize=(7, 4))
    fig.clf()

    curvelinear_test1(fig)
    plt.show()







********************************************
./axes_grid/demo_axes_divider.py
import matplotlib.pyplot as plt

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)


def demo_simple_image(ax):
    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    cb = plt.colorbar(im)
    plt.setp(cb.ax.get_yticklabels(), visible=False)


def demo_locatable_axes_hard(fig1):

    from mpl_toolkits.axes_grid1 \
         import SubplotDivider, LocatableAxes, Size

    divider = SubplotDivider(fig1, 2, 2, 2, aspect=True)

    # axes for image
    ax = LocatableAxes(fig1, divider.get_position())

    # axes for colorbar
    ax_cb = LocatableAxes(fig1, divider.get_position())

    h = [Size.AxesX(ax), # main axes
         Size.Fixed(0.05), # padding, 0.1 inch
         Size.Fixed(0.2), # colorbar, 0.3 inch
         ]

    v = [Size.AxesY(ax)]

    divider.set_horizontal(h)
    divider.set_vertical(v)

    ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    ax_cb.set_axes_locator(divider.new_locator(nx=2, ny=0))

    fig1.add_axes(ax)
    fig1.add_axes(ax_cb)

    ax_cb.axis["left"].toggle(all=False)
    ax_cb.axis["right"].toggle(ticks=True)

    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    plt.colorbar(im, cax=ax_cb)
    plt.setp(ax_cb.get_yticklabels(), visible=False)


def demo_locatable_axes_easy(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax_cb)

    Z, extent = get_demo_image()
    im = ax.imshow(Z, extent=extent, interpolation="nearest")

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    for tl in ax_cb.get_yticklabels():
        tl.set_visible(False)
    ax_cb.yaxis.tick_right()


def demo_images_side_by_side(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    Z, extent = get_demo_image()
    ax2 = divider.new_horizontal(size="100%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)

    ax.imshow(Z, extent=extent, interpolation="nearest")
    ax2.imshow(Z, extent=extent, interpolation="nearest")
    for tl in ax2.get_yticklabels():
        tl.set_visible(False)


def demo():

    fig1 = plt.figure(1, (6, 6))
    fig1.clf()

    ## PLOT 1
    # simple image & colorbar
    ax = fig1.add_subplot(2, 2, 1)
    demo_simple_image(ax)

    ## PLOT 2
    # image and colorbar whose location is adjusted in the drawing time.
    # a hard way

    demo_locatable_axes_hard(fig1)


    ## PLOT 3
    # image and colorbar whose location is adjusted in the drawing time.
    # a easy way

    ax = fig1.add_subplot(2, 2, 3)
    demo_locatable_axes_easy(ax)


    ## PLOT 4
    # two images side by side with fixed padding.

    ax = fig1.add_subplot(2, 2, 4)
    demo_images_side_by_side(ax)

    plt.draw()
    plt.show()


demo()




********************************************
./axes_grid/demo_axisline_style.py

from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
import numpy as np

if 1:
    fig = plt.figure(1)
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

    x = np.linspace(-0.5, 1., 100)
    ax.plot(x, np.sin(x*np.pi))

    plt.show()




********************************************
./axes_grid/demo_axes_grid2.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def get_demo_image():
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

if 1:
    F = plt.figure(1, (6, 6))
    F.clf()

    # prepare images
    Z, extent = get_demo_image()
    ZS = [Z[i::3,:] for i in range(3)]
    extent = extent[0], extent[1]/3., extent[2], extent[3]

    # demo 1 : colorbar at each axes

    grid = ImageGrid(F, 211, # similar to subplot(111)
                    nrows_ncols = (1, 3),
                    direction="row",
                    axes_pad = 0.05,
                    add_all=True,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="top",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="1%",
                    )


    for ax, z in zip(grid, ZS):
        im = ax.imshow(z, origin="lower", extent=extent, interpolation="nearest")
        ax.cax.colorbar(im)

    for ax, im_title in zip(grid, ["Image 1", "Image 2", "Image 3"]):
        t = add_inner_title(ax, im_title, loc=3)
        t.patch.set_alpha(0.5)

    for ax, z in zip(grid, ZS):
        ax.cax.toggle_label(True)
        #axis = ax.cax.axis[ax.cax.orientation]
        #axis.label.set_text("counts s$^{-1}$")
        #axis.label.set_size(10)
        #axis.major_ticklabels.set_size(6)

    # changing the colorbar ticks
    grid[1].cax.set_xticks([-1, 0, 1])
    grid[2].cax.set_xticks([-1, 0, 1])

    grid[0].set_xticks([-2, 0])
    grid[0].set_yticks([-2, 0, 2])


    # demo 2 : shared colorbar

    grid2 = ImageGrid(F, 212,
                      nrows_ncols = (1, 3),
                      direction="row",
                      axes_pad = 0.05,
                      add_all=True,
                      label_mode = "1",
                      share_all = True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="10%",
                      cbar_pad=0.05,
                      )

    grid2[0].set_xlabel("X")
    grid2[0].set_ylabel("Y")

    vmax, vmin = np.max(ZS), np.min(ZS)
    import matplotlib.colors
    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)

    for ax, z in zip(grid2, ZS):
        im = ax.imshow(z, norm=norm,
                       origin="lower", extent=extent,
                       interpolation="nearest")

    # With cbar_mode="single", cax attribute of all axes are identical.
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    for ax, im_title in zip(grid2, ["(a)", "(b)", "(c)"]):
        t = add_inner_title(ax, im_title, loc=2)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    grid2[0].set_xticks([-2, 0])
    grid2[0].set_yticks([-2, 0, 2])

    plt.draw()
    plt.show()




********************************************
./axes_grid/inset_locator_demo2.py
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np

def get_demo_image():
    from matplotlib.cbook import get_sample_data
    import numpy as np
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)

fig, ax = plt.subplots(figsize=[5,4])

# prepare the demo image
Z, extent = get_demo_image()
Z2 = np.zeros([150, 150], dtype="d")
ny, nx = Z.shape
Z2[30:30+ny, 30:30+nx] = Z

# extent = [-3, 4, -4, 3]
ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
             origin="lower")

# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.show()





********************************************
./axes_grid/demo_axes_grid.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)

def demo_simple_grid(fig):
    """
    A grid of 2x2 images with 0.05 inch pad between images and only
    the lower-left axes is labeled.
    """
    grid = AxesGrid(fig, 131, # similar to subplot(131)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.05,
                    label_mode = "1",
                    )

    Z, extent = get_demo_image()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")

    # This only affects axes in first column and second row as share_all = False.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


def demo_grid_with_single_cbar(fig):
    """
    A grid of 2x2 images with a single colorbar
    """
    grid = AxesGrid(fig, 132, # similar to subplot(132)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.0,
                    share_all=True,
                    label_mode = "L",
                    cbar_location = "top",
                    cbar_mode="single",
                    )

    Z, extent = get_demo_image()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
    #plt.colorbar(im, cax = grid.cbar_axes[0])
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)
        
    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


def demo_grid_with_each_cbar(fig):
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """

    grid = AxesGrid(F, 133, # similar to subplot(122)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.1,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="top",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    Z, extent = get_demo_image()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
        grid.cbar_axes[i].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])



if 1:
    F = plt.figure(1, (5.5, 2.5))

    F.subplots_adjust(left=0.05, right=0.98)

    demo_simple_grid(F)
    demo_grid_with_single_cbar(F)
    demo_grid_with_each_cbar(F)

    plt.draw()
    plt.show()





********************************************
./axes_grid/demo_colorbar_with_inset_locator.py
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])

axins1 = inset_axes(ax1,
                    width="50%", # width = 10% of parent_bbox width
                    height="5%", # height : 50%
                    loc=1)

im1=ax1.imshow([[1,2],[2, 3]])
plt.colorbar(im1, cax=axins1, orientation="horizontal", ticks=[1,2,3])
axins1.xaxis.set_ticks_position("bottom")

axins = inset_axes(ax2,
                   width="5%", # width = 10% of parent_bbox width
                   height="50%", # height : 50%
                   loc=3,
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax2.transAxes,
                   borderpad=0,
                   )

# Controlling the placement of the inset axes is basically same as that
# of the legend.  you may want to play with the borderpad value and
# the bbox_to_anchor coordinate.

im=ax2.imshow([[1,2],[2, 3]])
plt.colorbar(im, cax=axins, ticks=[1,2,3])

plt.draw()
plt.show()




********************************************
./misc/rec_join_demo.py
from __future__ import print_function
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('aapl.csv', asfileobj=False)
print('loading', datafile)
r = mlab.csv2rec(datafile)

r.sort()
r1 = r[-10:]

# Create a new array
r2 = np.empty(12, dtype=[('date', '|O4'), ('high', np.float),
                            ('marker', np.float)])
r2 = r2.view(np.recarray)
r2.date = r.date[-17:-5]
r2.high = r.high[-17:-5]
r2.marker = np.arange(12)

print("r1:")
print(mlab.rec2txt(r1))
print("r2:")
print(mlab.rec2txt(r2))

defaults = {'marker':-1, 'close':np.NaN, 'low':-4444.}

for s in ('inner', 'outer', 'leftouter'):
    rec = mlab.rec_join(['date', 'high'], r1, r2,
            jointype=s, defaults=defaults)
    print("\n%sjoin :\n%s" % (s, mlab.rec2txt(rec)))




********************************************
./misc/image_thumbnail.py
"""
You can use matplotlib to generate thumbnails from existing images.
matplotlib natively supports PNG files on the input side, and other
image types transparently if your have PIL installed
"""

from __future__ import print_function
# build thumbnails of all images in a directory
import sys, os, glob
import matplotlib.image as image


if len(sys.argv)!=2:
    print('Usage: python %s IMAGEDIR'%__file__)
    raise SystemExit
indir = sys.argv[1]
if not os.path.isdir(indir):
    print('Could not find input directory "%s"'%indir)
    raise SystemExit

outdir = 'thumbs'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for fname in glob.glob(os.path.join(indir, '*.png')):
    basedir, basename = os.path.split(fname)
    outfile = os.path.join(outdir, basename)
    fig = image.thumbnail(fname, outfile, scale=0.15)
    print('saved thumbnail of %s to %s'%(fname, outfile))





********************************************
./misc/font_indexing.py
"""
A little example that shows how the various indexing into the font
tables relate to one another.  Mainly for mpl developers....

"""
from __future__ import print_function
import matplotlib
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT, KERNING_UNFITTED, KERNING_UNSCALED



#fname = '/usr/share/fonts/sfd/FreeSans.ttf'
fname = matplotlib.get_data_path() + '/fonts/ttf/Vera.ttf'
font = FT2Font(fname)
font.set_charmap(0)

codes = font.get_charmap().items()
#dsu = [(ccode, glyphind) for ccode, glyphind in codes]
#dsu.sort()
#for ccode, glyphind in dsu:
#    try: name = font.get_glyph_name(glyphind)
#    except RuntimeError: pass
#    else: print '% 4d % 4d %s %s'%(glyphind, ccode, hex(int(ccode)), name)



# make a charname to charcode and glyphind dictionary
coded = {}
glyphd = {}
for ccode, glyphind in codes:
    name = font.get_glyph_name(glyphind)
    coded[name] = ccode
    glyphd[name] = glyphind

code =  coded['A']
glyph = font.load_char(code)
#print glyph.bbox
print(glyphd['A'], glyphd['V'], coded['A'], coded['V'])
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_DEFAULT))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNFITTED))
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNSCALED))
print('AV', font.get_kerning(glyphd['A'], glyphd['T'], KERNING_UNSCALED))




********************************************
./misc/sample_data_demo.py
"""
Grab mpl data from the ~/.matplotlib/sample_data cache if it exists, else
fetch it from github and cache it
"""
from __future__ import print_function
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
fname = cbook.get_sample_data('ada.png', asfileobj=False)

print('fname', fname)
im = plt.imread(fname)
plt.imshow(im)
plt.show()




********************************************
./misc/contour_manual.py
"""
Example of displaying your own contour lines and polygons using ContourSet.
"""
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
import matplotlib.cm as cm

# Contour lines for each level are a list/tuple of polygons.
lines0 = [ [[0,0],[0,4]] ]
lines1 = [ [[2,0],[1,2],[1,3]] ]
lines2 = [ [[3,0],[3,2]], [[3,3],[3,4]] ]  # Note two lines.

# Filled contours between two levels are also a list/tuple of polygons.
# Points can be ordered clockwise or anticlockwise.
filled01 = [ [[0,0],[0,4],[1,3],[1,2],[2,0]] ]
filled12 = [ [[2,0],[3,0],[3,2],[1,3],[1,2]],   # Note two polygons.
             [[1,4],[3,4],[3,3]] ]


plt.figure()

# Filled contours using filled=True.
cs = ContourSet(plt.gca(), [0,1,2], [filled01, filled12], filled=True, cmap=cm.bone)
cbar = plt.colorbar(cs)

# Contour lines (non-filled).
lines = ContourSet(plt.gca(), [0,1,2], [lines0, lines1, lines2], cmap=cm.cool,
                   linewidths=3)
cbar.add_lines(lines)

plt.axis([-0.5, 3.5, -0.5, 4.5])
plt.title('User-specified contours')



# Multiple filled contour lines can be specified in a single list of polygon
# vertices along with a list of vertex kinds (code types) as described in the
# Path class.  This is particularly useful for polygons with holes.
# Here a code type of 1 is a MOVETO, and 2 is a LINETO.

plt.figure()
filled01 = [ [[0,0],[3,0],[3,3],[0,3],[1,1],[1,2],[2,2],[2,1]] ]
kinds01  = [ [1,2,2,2,1,2,2,2] ]
cs = ContourSet(plt.gca(), [0,1], [filled01], [kinds01], filled=True)
cbar = plt.colorbar(cs)

plt.axis([-0.5, 3.5, -0.5, 3.5])
plt.title('User specified filled contours with holes')

plt.show()



********************************************
./misc/multiprocess.py
#Demo of using multiprocessing for generating data in one process and plotting
#in another.
#Written by Robert Cimrman
#Requires >= Python 2.6 for the multiprocessing module or having the
#standalone processing module installed

from __future__ import print_function
import time
try:
    from multiprocessing import Process, Pipe
except ImportError:
    from processing import Process, Pipe
import numpy as np

import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import gobject

class ProcessPlotter(object):

    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def poll_draw(self):

        def call_back():
            while 1:
                if not self.pipe.poll():
                    break

                command = self.pipe.recv()

                if command is None:
                    self.terminate()
                    return False

                else:
                    self.x.append(command[0])
                    self.y.append(command[1])
                    self.ax.plot(self.x, self.y, 'ro')

            self.fig.canvas.draw()
            return True

        return call_back

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        self.gid = gobject.timeout_add(1000, self.poll_draw())

        print('...done')
        plt.show()


class NBPlot(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = Process(target = self.plotter,
                                    args = (plotter_pipe,))
        self.plot_process.daemon = True
        self.plot_process.start()

    def plot(self, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            data = np.random.random(2)
            send(data)

def main():
    pl = NBPlot()
    for ii in range(10):
        pl.plot()
        time.sleep(0.5)
    raw_input('press Enter...')
    pl.plot(finished=True)

if __name__ == '__main__':
    main()




********************************************
./misc/longshort.py
"""
Illustrate the rec array utility funcitons by loading prices from a
csv file, computing the daily returns, appending the results to the
record arrays, joining on date
"""
import urllib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# grab the price data off yahoo
u1 = urllib.urlretrieve('http://ichart.finance.yahoo.com/table.csv?s=AAPL&d=9&e=14&f=2008&g=d&a=8&b=7&c=1984&ignore=.csv')
u2 = urllib.urlretrieve('http://ichart.finance.yahoo.com/table.csv?s=GOOG&d=9&e=14&f=2008&g=d&a=8&b=7&c=1984&ignore=.csv')

# load the CSV files into record arrays
r1 = mlab.csv2rec(file(u1[0]))
r2 = mlab.csv2rec(file(u2[0]))

# compute the daily returns and add these columns to the arrays
gains1 = np.zeros_like(r1.adj_close)
gains2 = np.zeros_like(r2.adj_close)
gains1[1:] = np.diff(r1.adj_close)/r1.adj_close[:-1]
gains2[1:] = np.diff(r2.adj_close)/r2.adj_close[:-1]
r1 = mlab.rec_append_fields(r1, 'gains', gains1)
r2 = mlab.rec_append_fields(r2, 'gains', gains2)

# now join them by date; the default postfixes are 1 and 2.  The
# default jointype is inner so it will do an intersection of dates and
# drop the dates in AAPL which occurred before GOOG started trading in
# 2004.  r1 and r2 are reverse ordered by date since Yahoo returns
# most recent first in the CSV files, but rec_join will sort by key so
# r below will be properly sorted
r = mlab.rec_join('date', r1, r2)


# long appl, short goog
g = r.gains1-r.gains2
tr = (1+g).cumprod()  # the total return

# plot the return
fig, ax = plt.subplots()
ax.plot(r.date, tr)
ax.set_title('total return: long APPL, short GOOG')
ax.grid()
fig.autofmt_xdate()
plt.show()




********************************************
./misc/rasterization_demo.py
import numpy as np
import matplotlib.pyplot as plt

d = np.arange(100).reshape(10, 10)
x, y = np.meshgrid(np.arange(11), np.arange(11))

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)
yy = x*np.sin(theta) + y*np.cos(theta)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.set_aspect(1)
ax1.pcolormesh(xx, yy, d)
ax1.set_title("No Rasterization")

ax2.set_aspect(1)
ax2.set_title("Rasterization")

m = ax2.pcolormesh(xx, yy, d)
m.set_rasterized(True)

ax3.set_aspect(1)
ax3.pcolormesh(xx, yy, d)
ax3.text(0.5, 0.5, "Text", alpha=0.2,
         va="center", ha="center", size=50, transform=ax3.transAxes)

ax3.set_title("No Rasterization")


ax4.set_aspect(1)
m = ax4.pcolormesh(xx, yy, d)
m.set_zorder(-20)

ax4.text(0.5, 0.5, "Text", alpha=0.2,
         zorder=-15,
         va="center", ha="center", size=50, transform=ax4.transAxes)

ax4.set_rasterization_zorder(-10)

ax4.set_title("Rasterization z$<-10$")


# ax2.title.set_rasterized(True) # should display a warning

plt.savefig("test_rasterization.pdf", dpi=150)
plt.savefig("test_rasterization.eps", dpi=150)

if not plt.rcParams["text.usetex"]:
    plt.savefig("test_rasterization.svg", dpi=150)
    # svg backend currently ignores the dpi




********************************************
./misc/rc_traits.py
# Here is some example code showing how to define some representative
# rc properties and construct a matplotlib artist using traits.
# matplotlib does not ship with enthought.traits, so you will need to
# install it separately.

from __future__ import print_function

import sys, os, re
import traits.api as traits
from matplotlib.cbook import is_string_like
from matplotlib.artist import Artist

doprint = True
flexible_true_trait = traits.Trait(
   True,
   { 'true':  True, 't': True, 'yes': True, 'y': True, 'on':  True, True: True,
     'false': False, 'f': False, 'no':  False, 'n': False, 'off': False, False: False
                              } )
flexible_false_trait = traits.Trait( False, flexible_true_trait )

colors = {
   'c' : '#00bfbf',
   'b' : '#0000ff',
   'g' : '#008000',
   'k' : '#000000',
   'm' : '#bf00bf',
   'r' : '#ff0000',
   'w' : '#ffffff',
   'y' : '#bfbf00',
   'gold'                 : '#FFD700',
   'peachpuff'            : '#FFDAB9',
   'navajowhite'          : '#FFDEAD',
   }

def hex2color(s):
   "Convert hex string (like html uses, eg, #efefef) to a r,g,b tuple"
   return tuple([int(n, 16)/255.0 for n in (s[1:3], s[3:5], s[5:7])])

class RGBA(traits.HasTraits):
   # r,g,b,a in the range 0-1 with default color 0,0,0,1 (black)
   r = traits.Range(0., 1., 0.)
   g = traits.Range(0., 1., 0.)
   b = traits.Range(0., 1., 0.)
   a = traits.Range(0., 1., 1.)
   def __init__(self, r=0., g=0., b=0., a=1.):
       self.r = r
       self.g = g
       self.b = b
       self.a = a
   def __repr__(self):
       return 'r,g,b,a = (%1.2f, %1.2f, %1.2f, %1.2f)'%\
              (self.r, self.g, self.b, self.a)

def tuple_to_rgba(ob, name, val):
   tup = [float(x) for x in val]
   if len(tup)==3:
       r,g,b = tup
       return RGBA(r,g,b)
   elif len(tup)==4:
       r,g,b,a = tup
       return RGBA(r,g,b,a)
   else:
       raise ValueError
tuple_to_rgba.info = 'a RGB or RGBA tuple of floats'

def hex_to_rgba(ob, name, val):
   rgx = re.compile('^#[0-9A-Fa-f]{6}$')

   if not is_string_like(val):
       raise TypeError
   if rgx.match(val) is None:
       raise ValueError
   r,g,b = hex2color(val)
   return RGBA(r,g,b,1.0)
hex_to_rgba.info = 'a hex color string'

def colorname_to_rgba(ob, name, val):
   hex = colors[val.lower()]
   r,g,b =  hex2color(hex)
   return RGBA(r,g,b,1.0)
colorname_to_rgba.info = 'a named color'

def float_to_rgba(ob, name, val):
   val = float(val)
   return RGBA(val, val, val, 1.)
float_to_rgba.info = 'a grayscale intensity'



Color = traits.Trait(RGBA(), float_to_rgba, colorname_to_rgba, RGBA,
             hex_to_rgba, tuple_to_rgba)

def file_exists(ob, name, val):
   fh = file(val, 'r')
   return val

def path_exists(ob, name, val):
   os.path.exists(val)
linestyles  = ('-', '--', '-.', ':', 'steps', 'None')
TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN = range(4)
linemarkers = (None, '.', ',', 'o', '^', 'v', '<', '>', 's',
                 '+', 'x', 'd', 'D', '|', '_', 'h', 'H',
                 'p', '1', '2', '3', '4',
                 TICKLEFT,
                 TICKRIGHT,
                 TICKUP,
                 TICKDOWN,
                 'None'
              )

class LineRC(traits.HasTraits):
   linewidth       = traits.Float(0.5)
   linestyle       = traits.Trait(*linestyles)
   color           = Color
   marker          = traits.Trait(*linemarkers)
   markerfacecolor = Color
   markeredgecolor = Color
   markeredgewidth = traits.Float(0.5)
   markersize      = traits.Float(6)
   antialiased     = flexible_true_trait
   data_clipping   = flexible_false_trait

class PatchRC(traits.HasTraits):
   linewidth       = traits.Float(1.0)
   facecolor = Color
   edgecolor = Color
   antialiased     = flexible_true_trait

timezones = 'UTC', 'US/Central', 'ES/Eastern' # fixme: and many more
backends = ('GTKAgg', 'Cairo', 'GDK', 'GTK', 'Agg',
           'GTKCairo', 'PS', 'SVG', 'Template', 'TkAgg',
           'WX')

class RC(traits.HasTraits):
   backend = traits.Trait(*backends)
   interactive  = flexible_false_trait
   toolbar      = traits.Trait('toolbar2', 'classic', None)
   timezone     = traits.Trait(*timezones)
   lines        = traits.Trait(LineRC())
   patch        = traits.Trait(PatchRC())

rc = RC()
rc.lines.color = 'r'
if doprint:
   print('RC')
   rc.print_traits()
   print('RC lines')
   rc.lines.print_traits()
   print('RC patches')
   rc.patch.print_traits()


class Patch(Artist, traits.HasTraits):
   linewidth = traits.Float(0.5)
   facecolor = Color
   fc = facecolor
   edgecolor = Color
   fill = flexible_true_trait
   def __init__(self,
                edgecolor=None,
                facecolor=None,
                linewidth=None,
                antialiased = None,
                fill=1,
                **kwargs
                ):
       Artist.__init__(self)

       if edgecolor is None: edgecolor = rc.patch.edgecolor
       if facecolor is None: facecolor = rc.patch.facecolor
       if linewidth is None: linewidth = rc.patch.linewidth
       if antialiased is None: antialiased = rc.patch.antialiased

       self.edgecolor = edgecolor
       self.facecolor = facecolor
       self.linewidth = linewidth
       self.antialiased = antialiased
       self.fill = fill


p = Patch()
p.facecolor = '#bfbf00'
p.edgecolor = 'gold'
p.facecolor = (1,.5,.5,.25)
p.facecolor = 0.25
p.fill = 'f'
print('p.facecolor', type(p.facecolor), p.facecolor)
print('p.fill', type(p.fill), p.fill)
if p.fill_: print('fill')
else: print('no fill')
if doprint:
   print()
   print('Patch')
   p.print_traits()




********************************************
./misc/ftface_props.py
#!/usr/bin/env python

from __future__ import print_function
"""
This is a demo script to show you how to use all the properties of an
FT2Font object.  These describe global font properties.  For
individual character metrices, use the Glyp object, as returned by
load_char
"""
import matplotlib
from matplotlib.ft2font import FT2Font

#fname = '/usr/local/share/matplotlib/VeraIt.ttf'
fname = matplotlib.get_data_path() + '/fonts/ttf/VeraIt.ttf'
#fname = '/usr/local/share/matplotlib/cmr10.ttf'

font = FT2Font(fname)

# these constants are used to access the style_flags and face_flags
FT_FACE_FLAG_SCALABLE          = 1 << 0
FT_FACE_FLAG_FIXED_SIZES       = 1 << 1
FT_FACE_FLAG_FIXED_WIDTH       = 1 << 2
FT_FACE_FLAG_SFNT              = 1 << 3
FT_FACE_FLAG_HORIZONTAL        = 1 << 4
FT_FACE_FLAG_VERTICAL          = 1 << 5
FT_FACE_FLAG_KERNING           = 1 << 6
FT_FACE_FLAG_FAST_GLYPHS       = 1 << 7
FT_FACE_FLAG_MULTIPLE_MASTERS  = 1 << 8
FT_FACE_FLAG_GLYPH_NAMES       = 1 << 9
FT_FACE_FLAG_EXTERNAL_STREAM   = 1 << 10
FT_STYLE_FLAG_ITALIC           = 1 << 0
FT_STYLE_FLAG_BOLD             = 1 << 1

print('Num faces   :', font.num_faces)       # number of faces in file
print('Num glyphs  :', font.num_glyphs)      # number of glyphs in the face
print('Family name :', font.family_name)     # face family name
print('Syle name   :', font.style_name)      # face syle name
print('PS name     :', font.postscript_name) # the postscript name
print('Num fixed   :', font.num_fixed_sizes) # number of embedded bitmap in face

# the following are only available if face.scalable
if font.scalable:
    # the face global bounding box (xmin, ymin, xmax, ymax)
    print('Bbox                :', font.bbox)
    # number of font units covered by the EM
    print('EM                  :', font.units_per_EM)
    # the ascender in 26.6 units
    print('Ascender            :', font.ascender)
    # the descender in 26.6 units
    print('Descender           :', font.descender)
    # the height in 26.6 units
    print('Height              :', font.height)
    # maximum horizontal cursor advance
    print('Max adv width       :', font.max_advance_width)
    # same for vertical layout
    print('Max adv height      :', font.max_advance_height)
    # vertical position of the underline bar
    print('Underline pos       :', font.underline_position)
    # vertical thickness of the underline
    print('Underline thickness :', font.underline_thickness)

print('Italics       :', font.style_flags & FT_STYLE_FLAG_ITALIC          != 0)
print('Bold          :', font.style_flags & FT_STYLE_FLAG_BOLD            != 0)
print('Scalable      :', font.style_flags & FT_FACE_FLAG_SCALABLE         != 0)
print('Fixed sizes   :', font.style_flags & FT_FACE_FLAG_FIXED_SIZES      != 0)
print('Fixed width   :', font.style_flags & FT_FACE_FLAG_FIXED_WIDTH      != 0)
print('SFNT          :', font.style_flags & FT_FACE_FLAG_SFNT             != 0)
print('Horizontal    :', font.style_flags & FT_FACE_FLAG_HORIZONTAL       != 0)
print('Vertical      :', font.style_flags & FT_FACE_FLAG_VERTICAL         != 0)
print('Kerning       :', font.style_flags & FT_FACE_FLAG_KERNING          != 0)
print('Fast glyphs   :', font.style_flags & FT_FACE_FLAG_FAST_GLYPHS      != 0)
print('Mult. masters :', font.style_flags & FT_FACE_FLAG_MULTIPLE_MASTERS != 0)
print('Glyph names   :', font.style_flags & FT_FACE_FLAG_GLYPH_NAMES      != 0)

print(dir(font))

cmap = font.get_charmap()
print(font.get_kerning)




********************************************
./misc/tight_bbox_test.py
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes([0.1, 0.3, 0.5, 0.5])

ax.pcolormesh(np.array([[1,2],[3,4]]))
plt.yticks([0.5, 1.5], ["long long tick label",
                        "tick label"])
plt.ylabel("My y-label")
plt.title("Check saved figures for their bboxes")
for ext in ["png", "pdf", "svg", "svgz", "eps"]:
    print("saving tight_bbox_test.%s" % (ext,))
    plt.savefig("tight_bbox_test.%s" % (ext,), bbox_inches="tight")
plt.show()




********************************************
./misc/rec_groupby_demo.py
from __future__ import print_function
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('aapl.csv', asfileobj=False)
print('loading', datafile)
r = mlab.csv2rec(datafile)
r.sort()

def daily_return(prices):
    'an array of daily returns from price array'
    g = np.zeros_like(prices)
    g[1:] = (prices[1:]-prices[:-1])/prices[:-1]
    return g

def volume_code(volume):
    'code the continuous volume data categorically'
    ind = np.searchsorted([1e5,1e6, 5e6,10e6, 1e7], volume)
    return ind

# a list of (dtype_name, summary_function, output_dtype_name).
# rec_summarize will call on each function on the indicated recarray
# attribute, and the result assigned to output name in the return
# record array.
summaryfuncs = (
    ('date', lambda x: [thisdate.year for thisdate in x], 'years'),
    ('date', lambda x: [thisdate.month for thisdate in x], 'months'),
    ('date', lambda x: [thisdate.weekday() for thisdate in x], 'weekday'),
    ('adj_close', daily_return, 'dreturn'),
    ('volume', volume_code, 'volcode'),
    )

rsum = mlab.rec_summarize(r, summaryfuncs)

# stats is a list of (dtype_name, function, output_dtype_name).
# rec_groupby will summarize the attribute identified by the
# dtype_name over the groups in the groupby list, and assign the
# result to the output_dtype_name
stats = (
    ('dreturn', len, 'rcnt'),
    ('dreturn', np.mean, 'rmean'),
    ('dreturn', np.median, 'rmedian'),
    ('dreturn', np.std, 'rsigma'),
    )

# you can summarize over a single variable, like years or months
print('summary by years')
ry = mlab.rec_groupby(rsum, ('years',), stats)
print(mlab. rec2txt(ry))

print('summary by months')
rm = mlab.rec_groupby(rsum, ('months',), stats)
print(mlab.rec2txt(rm))

# or over multiple variables like years and months
print('summary by year and month')
rym = mlab.rec_groupby(rsum, ('years','months'), stats)
print(mlab.rec2txt(rym))

print('summary by volume')
rv = mlab.rec_groupby(rsum, ('volcode',), stats)
print(mlab.rec2txt(rv))




********************************************
./misc/svg_filter_pie.py
"""
Demonstrate SVG filtering effects which might be used with mpl.
The pie chart drawing code is borrowed from pie_demo.py

Note that the filtering effects are only effective if your svg rederer
support it. 
"""


import matplotlib
matplotlib.use("Svg")

import matplotlib.pyplot as plt
from matplotlib.patches import Shadow

# make a square figure and axes
fig1 = plt.figure(1, figsize=(6,6))
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15,30,45, 10]

explode=(0, 0.05, 0, 0)

# We want to draw the shadow for each pie but we will not use "shadow"
# option as it does'n save the references to the shadow patches.
pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')

for w in pies[0]:
    # set the id with the label.
    w.set_gid(w.get_label())

    # we don't want to draw the edge of the pie
    w.set_ec("none")

for w in pies[0]:
    # create shadow patch
    s = Shadow(w, -0.01, -0.01)
    s.set_gid(w.get_gid()+"_shadow")
    s.set_zorder(w.get_zorder() - 0.1)
    ax.add_patch(s)
    

# save
from StringIO import StringIO
f = StringIO()
plt.savefig(f, format="svg")

import xml.etree.cElementTree as ET


# filter definition for shadow using a gaussian blur
# and lighteneing effect.
# The lightnening filter is copied from http://www.w3.org/TR/SVG/filters.html

# I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
# in both, but the lightnening effect only in the inkscape. Also note
# that, inkscape's exporting also may not support it.

filter_def = """
  <defs  xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>
    <filter id='dropshadow' height='1.2' width='1.2'>
      <feGaussianBlur result='blur' stdDeviation='2'/>
    </filter>
    
    <filter id='MyFilter' filterUnits='objectBoundingBox' x='0' y='0' width='1' height='1'>
      <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
      <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
      <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75' 
           specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
        <fePointLight x='-5000%' y='-10000%' z='20000%'/>
      </feSpecularLighting>
      <feComposite in='specOut' in2='SourceAlpha' operator='in' result='specOut'/>
      <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic' 
    k1='0' k2='1' k3='1' k4='0'/>
    </filter>
  </defs>
"""


tree, xmlid = ET.XMLID(f.getvalue())

# insert the filter definition in the svg dom tree.
tree.insert(0, ET.XML(filter_def))

for i, pie_name in enumerate(labels):
    pie = xmlid[pie_name]
    pie.set("filter", 'url(#MyFilter)')

    shadow = xmlid[pie_name + "_shadow"]
    shadow.set("filter",'url(#dropshadow)')

fn = "svg_filter_pie.svg"
print "Saving '%s'" % fn
ET.ElementTree(tree).write(fn)




********************************************
./misc/svg_filter_line.py
"""
Demonstrate SVG filtering effects which might be used with mpl.

Note that the filtering effects are only effective if your svg rederer
support it.
"""

from __future__ import print_function
import matplotlib

matplotlib.use("Svg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

fig1 = plt.figure()
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])

# draw lines
l1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-",
              mec="b", lw=5, ms=10, label="Line 1")
l2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "rs-",
              mec="r", lw=5, ms=10, color="r", label="Line 2")


for l in [l1, l2]:

    # draw shadows with same lines with slight offset and gray colors.

    xx = l.get_xdata()
    yy = l.get_ydata()
    shadow, = ax.plot(xx, yy)
    shadow.update_from(l)

    # adjust color
    shadow.set_color("0.2")
    # adjust zorder of the shadow lines so that it is drawn below the
    # original lines
    shadow.set_zorder(l.get_zorder()-0.5)

    # offset transform
    ot = mtransforms.offset_copy(l.get_transform(), fig1,
                                 x=4.0, y=-6.0, units='points')

    shadow.set_transform(ot)

    # set the id for a later use
    shadow.set_gid(l.get_label()+"_shadow")


ax.set_xlim(0., 1.)
ax.set_ylim(0., 1.)

# save the figure as a string in the svg format.
from StringIO import StringIO
f = StringIO()
plt.savefig(f, format="svg")


import xml.etree.cElementTree as ET

# filter definition for a gaussian blur
filter_def = """
  <defs  xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>
    <filter id='dropshadow' height='1.2' width='1.2'>
      <feGaussianBlur result='blur' stdDeviation='3'/>
    </filter>
  </defs>
"""


# read in the saved svg
tree, xmlid = ET.XMLID(f.getvalue())

# insert the filter definition in the svg dom tree.
tree.insert(0, ET.XML(filter_def))

for l in [l1, l2]:
    # pick up the svg element with given id
    shadow = xmlid[l.get_label()+"_shadow"]
    # apply shdow filter
    shadow.set("filter",'url(#dropshadow)')

fn = "svg_filter_line.svg"
print("Saving '%s'" % fn)
ET.ElementTree(tree).write(fn)




********************************************
./misc/developer_commit_history.py
from __future__ import print_function
"""
report how many days it has been since each developer committed.  You
must do an

svn log > log.txt

and place the output next to this file before running

"""
import os, datetime

import matplotlib.cbook as cbook

todate = cbook.todate('%Y-%m-%d')
today = datetime.date.today()
if not os.path.exists('log.txt'):
    print('You must place the "svn log" output into a file "log.txt"')
    raise SystemExit

parse = False

lastd = dict()
for line in file('log.txt'):
    if line.startswith('--------'):
        parse = True
        continue

    if parse:
        parts = [part.strip() for part in line.split('|')]
        developer = parts[1]
        dateparts = parts[2].split(' ')
        ymd = todate(dateparts[0])


    if developer not in lastd:
        lastd[developer] = ymd

    parse = False

dsu = [((today - lastdate).days, developer) for developer, lastdate in lastd.items()]

dsu.sort()
for timedelta, developer in dsu:
    print('%s : %d'%(developer, timedelta))




********************************************
./tests/backend_driver.py
#!/usr/bin/env python

from __future__ import print_function, division
"""
This is used to drive many of the examples across the backends, for
regression testing, and comparing backend efficiency.

You can specify the backends to be tested either via the --backends
switch, which takes a comma-separated list, or as separate arguments,
e.g.

    python backend_driver.py agg ps

would test the agg and ps backends. If no arguments are given, a
default list of backends will be tested.

Interspersed with the backend arguments can be switches for the Python
interpreter executing the tests. If entering such arguments causes an
option parsing error with the driver script, separate them from driver
switches with a --.
"""

import os
import time
import sys
import glob
from optparse import OptionParser

import matplotlib.rcsetup as rcsetup
from matplotlib.cbook import Bunch, dedent


all_backends = list(rcsetup.all_backends)  # to leave the original list alone

# actual physical directory for each dir
dirs = dict(files=os.path.join('..', 'lines_bars_and_markers'),
            shapes=os.path.join('..', 'shapes_and_collections'),
            images=os.path.join('..', 'images_contours_and_fields'),
            pie=os.path.join('..', 'pie_and_polar_charts'),
            text=os.path.join('..', 'text_labels_and_annotations'),
            ticks=os.path.join('..', 'ticks_and_spines'),
            subplots=os.path.join('..', 'subplots_axes_and_figures'),
            specialty=os.path.join('..', 'specialty_plots'),
            showcase=os.path.join('..', 'showcase'),
            pylab = os.path.join('..', 'pylab_examples'),
            api = os.path.join('..', 'api'),
            units = os.path.join('..', 'units'),
            mplot3d = os.path.join('..', 'mplot3d'))


# files in each dir
files = dict()

files['lines'] = [
    'barh_demo.py',
    'fill_demo.py',
    'fill_demo_features.py',
    'line_demo_dash_control.py',
    ]

files['shapes'] = [
    'path_patch_demo.py',
    'scatter_demo.py',
    ]

files['colors'] = [
    'color_cycle_demo.py',
    ]

files['images'] = [
    'imshow_demo.py',
    ]

files['statistics'] = [
    'errorbar_demo.py',
    'errorbar_demo_features.py',
    'histogram_demo_features.py',
    ]

files['pie'] = [
    'pie_demo.py',
    'polar_bar_demo.py',
    'polar_scatter_demo.py',
    ]

files['text_labels_and_annotations'] = [
    'text_demo_fontdict.py',
    'unicode_demo.py',
    ]

files['ticks_and_spines'] = [
    'spines_demo_bounds.py',
    'ticklabels_demo_rotation.py',
    ]

files['subplots_axes_and_figures'] = [
    'subplot_demo.py',
    ]

files['showcase'] = [
    'integral_demo.py',
    ]

files['pylab'] = [
    'accented_text.py',
    'alignment_test.py',
    'annotation_demo.py',
    'annotation_demo.py',
    'annotation_demo2.py',
    'annotation_demo2.py',
    'anscombe.py',
    'arctest.py',
    'arrow_demo.py',
    'axes_demo.py',
    'axes_props.py',
    'axhspan_demo.py',
    'axis_equal_demo.py',
    'bar_stacked.py',
    'barb_demo.py',
    'barchart_demo.py',
    'barcode_demo.py',
    'boxplot_demo.py',
    'broken_barh.py',
    'clippedline.py',
    'cohere_demo.py',
    'color_by_yvalue.py',
    'color_demo.py',
    'colorbar_tick_labelling_demo.py',
    'contour_demo.py',
    'contour_image.py',
    'contour_label_demo.py',
    'contourf_demo.py',
    'contourf_log.py',
    'coords_demo.py',
    'coords_report.py',
    'csd_demo.py',
    'cursor_demo.py',
    'custom_cmap.py',
    'custom_figure_class.py',
    'custom_ticker1.py',
    'customize_rc.py',
    'dashpointlabel.py',
    'date_demo1.py',
    'date_demo2.py',
    'date_demo_convert.py',
    'date_demo_rrule.py',
    'date_index_formatter.py',
    'dolphin.py',
    'ellipse_collection.py',
    'ellipse_demo.py',
    'ellipse_rotated.py',
    'equal_aspect_ratio.py',
    'errorbar_limits.py',
    'fancyarrow_demo.py',
    'fancybox_demo.py',
    'fancybox_demo2.py',
    'fancytextbox_demo.py',
    'figimage_demo.py',
    'figlegend_demo.py',
    'figure_title.py',
    'fill_between_demo.py',
    'fill_spiral.py',
    'finance_demo.py',
    'findobj_demo.py',
    'fonts_demo.py',
    'fonts_demo_kw.py',
    'ganged_plots.py',
    'geo_demo.py',
    'gradient_bar.py',
    'griddata_demo.py',
    'hatch_demo.py',
    'hexbin_demo.py',
    'hexbin_demo2.py',
    'hist_colormapped.py',
    'histogram_demo_extended.py',
    'vline_hline_demo.py',

    'image_clip_path.py',
    'image_demo.py',
    'image_demo2.py',
    'image_interp.py',
    'image_masked.py',
    'image_nonuniform.py',
    'image_origin.py',
    'image_slices_viewer.py',
    'interp_demo.py',
    'invert_axes.py',
    'layer_images.py',
    'legend_auto.py',
    'legend_demo.py',
    'legend_demo2.py',
    'legend_demo3.py',
    'legend_scatter.py',
    'line_collection.py',
    'line_collection2.py',
    'line_styles.py',
    'log_bar.py',
    'log_demo.py',
    'log_test.py',
    'major_minor_demo1.py',
    'major_minor_demo2.py',
    'manual_axis.py',
    'masked_demo.py',
    'mathtext_demo.py',
    'mathtext_examples.py',
    'matplotlib_icon.py',
    'matshow.py',
    'mri_demo.py',
    'mri_with_eeg.py',
    'multi_image.py',
    'multiline.py',
    'multiple_figs_demo.py',
    'nan_test.py',
    'newscalarformatter_demo.py',
    'pcolor_demo.py',
    'pcolor_log.py',
    'pcolor_small.py',
    'pie_demo2.py',
    'plotfile_demo.py',
    'polar_demo.py',
    'polar_legend.py',
    'psd_demo.py',
    'psd_demo2.py',
    'psd_demo3.py',
    'quadmesh_demo.py',
    'quiver_demo.py',
    'scatter_custom_symbol.py',
    'scatter_demo2.py',
    'scatter_masked.py',
    'scatter_profile.py',
    'scatter_star_poly.py',
    #'set_and_get.py',
    'shared_axis_across_figures.py',
    'shared_axis_demo.py',
    'simple_plot.py',
    'specgram_demo.py',
    'spine_placement_demo.py',
    'spy_demos.py',
    'stem_plot.py',
    'step_demo.py',
    'stix_fonts_demo.py',
    'stock_demo.py',
    'subplots_adjust.py',
    'symlog_demo.py',
    'table_demo.py',
    'text_handles.py',
    'text_rotation.py',
    'text_rotation_relative_to_line.py',
    'transoffset.py',
    'xcorr_demo.py',
    'zorder_demo.py',
    ]


files['api'] = [
    'agg_oo.py',
    'barchart_demo.py',
    'bbox_intersect.py',
    'collections_demo.py',
    'colorbar_only.py',
    'custom_projection_example.py',
    'custom_scale_example.py',
    'date_demo.py',
    'date_index_formatter.py',
    'donut_demo.py',
    'font_family_rc.py',
    'image_zcoord.py',
    'joinstyle.py',
    'legend_demo.py',
    'line_with_text.py',
    'logo2.py',
    'mathtext_asarray.py',
    'patch_collection.py',
    'quad_bezier.py',
    'scatter_piecharts.py',
    'span_regions.py',
    'two_scales.py',
    'unicode_minus.py',
    'watermark_image.py',
    'watermark_text.py',
]

files['units'] = [
    'annotate_with_units.py',
    #'artist_tests.py',  # broken, fixme
    'bar_demo2.py',
    #'bar_unit_demo.py', # broken, fixme
    #'ellipse_with_units.py',  # broken, fixme
    'radian_demo.py',
    'units_sample.py',
    #'units_scatter.py', # broken, fixme

    ]

files['mplot3d'] = [
    '2dcollections3d_demo.py',
    'bars3d_demo.py',
    'contour3d_demo.py',
    'contour3d_demo2.py',
    'contourf3d_demo.py',
    'lines3d_demo.py',
    'polys3d_demo.py',
    'scatter3d_demo.py',
    'surface3d_demo.py',
    'surface3d_demo2.py',
    'text3d_demo.py',
    'wire3d_demo.py',
    ]

# dict from dir to files we know we don't want to test (eg examples
# not using pyplot, examples requiring user input, animation examples,
# examples that may only work in certain environs (usetex examples?),
# examples that generate multiple figures

excluded = {
    'pylab' : ['__init__.py', 'toggle_images.py',],
    'units' : ['__init__.py', 'date_support.py',],
}

def report_missing(dir, flist):
    'report the py files in dir that are not in flist'
    globstr = os.path.join(dir, '*.py')
    fnames = glob.glob(globstr)

    pyfiles = set([os.path.split(fullpath)[-1] for fullpath in set(fnames)])

    exclude = set(excluded.get(dir, []))
    flist = set(flist)
    missing = list(pyfiles-flist-exclude)
    missing.sort()
    if missing:
        print ('%s files not tested: %s'%(dir, ', '.join(missing)))

def report_all_missing(directories):
    for f in directories:
        report_missing(dirs[f], files[f])


# tests known to fail on a given backend

failbackend = dict(
    svg = ('tex_demo.py', ),
    agg = ('hyperlinks.py', ),
    pdf = ('hyperlinks.py', ),
    ps = ('hyperlinks.py', ),
    )


try:
    import subprocess
    def run(arglist):
        try:
            ret = subprocess.call(arglist)
        except KeyboardInterrupt:
            sys.exit()
        else:
            return ret
except ImportError:
    def run(arglist):
        os.system(' '.join(arglist))

def drive(backend, directories, python=['python'], switches = []):
    exclude = failbackend.get(backend, [])

    # Clear the destination directory for the examples
    path = backend
    if os.path.exists(path):
        import glob
        for fname in os.listdir(path):
            os.unlink(os.path.join(path, fname))
    else:
        os.mkdir(backend)
    failures = []

    testcases = [os.path.join(dirs[d], fname)
                 for d in directories
                 for fname in files[d]]

    for fullpath in testcases:
        print ('\tdriving %-40s' % (fullpath)),
        sys.stdout.flush()
        fpath, fname = os.path.split(fullpath)

        if fname in exclude:
            print ('\tSkipping %s, known to fail on backend: %s'%backend)
            continue

        basename, ext = os.path.splitext(fname)
        outfile = os.path.join(path, basename)
        tmpfile_name = '_tmp_%s.py' % basename
        tmpfile = open(tmpfile_name, 'w')

        future_imports = 'from __future__ import division, print_function'
        for line in open(fullpath):
            line_lstrip = line.lstrip()
            if line_lstrip.startswith("#"):
                tmpfile.write(line)
            elif 'unicode_literals' in line:
                future_imports = future_imports + ', unicode_literals'

        tmpfile.writelines((
            future_imports+'\n',
            'import sys\n',
            'sys.path.append("%s")\n' % fpath.replace('\\', '\\\\'),
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % backend,
            'from pylab import savefig\n',
            'import numpy\n',
            'numpy.seterr(invalid="ignore")\n',
            ))
        for line in open(fullpath):
            line_lstrip = line.lstrip()
            if (line_lstrip.startswith('from __future__ import') or
                line_lstrip.startswith('matplotlib.use') or
                line_lstrip.startswith('savefig') or
                line_lstrip.startswith('show')):
                continue
            tmpfile.write(line)
        if backend in rcsetup.interactive_bk:
            tmpfile.write('show()')
        else:
            tmpfile.write('\nsavefig(r"%s", dpi=150)' % outfile)

        tmpfile.close()
        start_time = time.time()
        program = [x % {'name': basename} for x in python]
        ret = run(program + [tmpfile_name] + switches)
        end_time = time.time()
        print ("%s %s" % ((end_time - start_time), ret))
        #os.system('%s %s %s' % (python, tmpfile_name, ' '.join(switches)))
        os.remove(tmpfile_name)
        if ret:
            failures.append(fullpath)
    return failures

def parse_options():
    doc = (__doc__ and __doc__.split('\n\n')) or "  "
    op = OptionParser(description=doc[0].strip(),
                      usage='%prog [options] [--] [backends and switches]',
                      #epilog='\n'.join(doc[1:])  # epilog not supported on my python2.4 machine: JDH
                      )
    op.disable_interspersed_args()
    op.set_defaults(dirs='pylab,api,units,mplot3d',
                    clean=False, coverage=False, valgrind=False)
    op.add_option('-d', '--dirs', '--directories', type='string',
                  dest='dirs', help=dedent('''
      Run only the tests in these directories; comma-separated list of
      one or more of: pylab (or pylab_examples), api, units, mplot3d'''))
    op.add_option('-b', '--backends', type='string', dest='backends',
                  help=dedent('''
      Run tests only for these backends; comma-separated list of
      one or more of: agg, ps, svg, pdf, template, cairo,
      Default is everything except cairo.'''))
    op.add_option('--clean', action='store_true', dest='clean',
                  help='Remove result directories, run no tests')
    op.add_option('-c', '--coverage', action='store_true', dest='coverage',
                  help='Run in coverage.py')
    op.add_option('-v', '--valgrind', action='store_true', dest='valgrind',
                  help='Run in valgrind')

    options, args = op.parse_args()
    switches = [x for x in args if x.startswith('--')]
    backends = [x.lower() for x in args if not x.startswith('--')]
    if options.backends:
        backends += [be.lower() for be in options.backends.split(',')]

    result = Bunch(
        dirs = options.dirs.split(','),
        backends = backends or ['agg', 'ps', 'svg', 'pdf', 'template'],
        clean = options.clean,
        coverage = options.coverage,
        valgrind = options.valgrind,
        switches = switches)
    if 'pylab_examples' in result.dirs:
        result.dirs[result.dirs.index('pylab_examples')] = 'pylab'
    #print result
    return (result)

if __name__ == '__main__':
    times = {}
    failures = {}
    options = parse_options()

    if options.clean:
        localdirs = [d for d in glob.glob('*') if os.path.isdir(d)]
        all_backends_set = set(all_backends)
        for d in localdirs:
            if d.lower() not in all_backends_set:
                continue
            print ('removing %s'%d)
            for fname in glob.glob(os.path.join(d, '*')):
                os.remove(fname)
            os.rmdir(d)
        for fname in glob.glob('_tmp*.py'):
            os.remove(fname)

        print ('all clean...')
        raise SystemExit
    if options.coverage:
        python = ['coverage.py', '-x']
    elif options.valgrind:
        python = ['valgrind', '--tool=memcheck', '--leak-check=yes',
                  '--log-file=%(name)s', sys.executable]
    elif sys.platform == 'win32':
        python = [sys.executable]
    else:
        python = [sys.executable]

    report_all_missing(options.dirs)
    for backend in options.backends:
        print ('testing %s %s' % (backend, ' '.join(options.switches)))
        t0 = time.time()
        failures[backend] = \
            drive(backend, options.dirs, python, options.switches)
        t1 = time.time()
        times[backend] = (t1-t0)/60.0

    # print times
    for backend, elapsed in times.items():
        print ('Backend %s took %1.2f minutes to complete' % (backend, elapsed))
        failed = failures[backend]
        if failed:
            print ('  Failures: %s' % failed)
        if 'template' in times:
            print ('\ttemplate ratio %1.3f, template residual %1.3f' % (
                elapsed/times['template'], elapsed-times['template']))




********************************************
./event_handling/pong_gtk.py
#!/usr/bin/env python

from __future__ import print_function

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import pipong


fig, ax = plt.subplots()
canvas = ax.figure.canvas


def start_anim(event):
#    gobject.idle_add(animation.draw,animation)
    gobject.timeout_add(10,animation.draw,animation)
    canvas.mpl_disconnect(start_anim.cid)

animation = pipong.Game(ax)
start_anim.cid = canvas.mpl_connect('draw_event', start_anim)


tstart = time.time()
plt.grid() # to ensure proper background restore
plt.show()
print('FPS:' , animation.cnt/(time.time()-tstart))




********************************************
./event_handling/close_event.py
from __future__ import print_function
import matplotlib.pyplot as plt

def handle_close(evt):
    print('Closed Figure!')

fig = plt.figure()
fig.canvas.mpl_connect('close_event', handle_close)

plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
plt.show()




********************************************
./event_handling/timers.py
# Simple example of using general timer objects. This is used to update
# the time placed in the title of the figure.
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def update_title(axes):
    axes.set_title(datetime.now())
    axes.figure.canvas.draw()

fig, ax = plt.subplots()

x = np.linspace(-3, 3)
ax.plot(x, x*x)

# Create a new timer object. Set the interval 500 milliseconds (1000 is default)
# and tell the timer what function should be called.
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(update_title, ax)
timer.start()

#Or could start the timer on first figure draw
#def start_timer(evt):
#    timer.start()
#    fig.canvas.mpl_disconnect(drawid)
#drawid = fig.canvas.mpl_connect('draw_event', start_timer)

plt.show()




********************************************
./event_handling/pick_event_demo2.py
"""
compute the mean and standard deviation (stddev) of 100 data sets and plot
mean vs stddev.  When you click on one of the mu, sigma points, plot the raw
data from the dataset that generated the mean and stddev.
"""
import numpy
import matplotlib.pyplot as plt


X = numpy.random.rand(100, 1000)
xs = numpy.mean(X, axis=1)
ys = numpy.std(X, axis=1)

fig, ax = plt.subplots()
ax.set_title('click on point to plot time series')
line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance


def onpick(event):

    if event.artist!=line: return True

    N = len(event.ind)
    if not N: return True


    figi = plt.figure()
    for subplotnum, dataind in enumerate(event.ind):
        ax = figi.add_subplot(N,1,subplotnum+1)
        ax.plot(X[dataind])
        ax.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),
                transform=ax.transAxes, va='top')
        ax.set_ylim(-0.5, 1.5)
    figi.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()









********************************************
./event_handling/trifinder_event_demo.py
"""
Example showing the use of a TriFinder object.  As the mouse is moved over the
triangulation, the triangle under the cursor is highlighted and the index of
the triangle is displayed in the plot title.
"""
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Polygon
import numpy as np
import math


def update_polygon(tri):
    if tri == -1:
        points = [0, 0, 0]
    else:
        points = triangulation.triangles[tri]
    xs = triangulation.x[points]
    ys = triangulation.y[points]
    polygon.set_xy(zip(xs, ys))


def motion_notify(event):
    if event.inaxes is None:
        tri = -1
    else:
        tri = trifinder(event.xdata, event.ydata)
    update_polygon(tri)
    plt.title('In triangle %i' % tri)
    event.canvas.draw()


# Create a Triangulation.
n_angles = 16
n_radii = 5
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)
angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += math.pi / n_angles
x = (radii*np.cos(angles)).flatten()
y = (radii*np.sin(angles)).flatten()
triangulation = Triangulation(x, y)
xmid = x[triangulation.triangles].mean(axis=1)
ymid = y[triangulation.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triangulation.set_mask(mask)

# Use the triangulation's default TriFinder object.
trifinder = triangulation.get_trifinder()

# Setup plot and callbacks.
plt.subplot(111, aspect='equal')
plt.triplot(triangulation, 'bo-')
polygon = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
update_polygon(-1)
plt.gca().add_patch(polygon)
plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify)
plt.show()




********************************************
./event_handling/path_editor.py
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()

pathdata = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
    ]

codes, verts = zip(*pathdata)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)


class PathInteractor:
    """
    An path editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them


    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, pathpatch):

        self.ax = pathpatch.axes
        canvas = self.ax.figure.canvas
        self.pathpatch = pathpatch
        self.pathpatch.set_animated(True)

        x, y = zip(*self.pathpatch.get_path().vertices)

        self.line, = ax.plot(x,y,marker='o', markerfacecolor='r', animated=True)

        self._ind = None # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas


    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def pathpatch_changed(self, pathpatch):
        'this method is called whenever the pathpatchgon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        plt.Artist.update_from(self.line, pathpatch)
        self.line.set_visible(vis)  # don't use the pathpatch visibility state


    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.pathpatch.get_path().vertices)
        xyt = self.pathpatch.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        ind = d.argmin()

        if d[ind]>=self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes==None: return
        if event.button != 1: return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes: return
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts: self._ind = None

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata

        vertices = self.pathpatch.get_path().vertices

        vertices[self._ind] = x,y
        self.line.set_data(zip(*vertices))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


interactor = PathInteractor(patch)
ax.set_title('drag vertices to update path')
ax.set_xlim(-3,4)
ax.set_ylim(-3,4)

plt.show()






********************************************
./event_handling/legend_picking.py
"""
Enable picking on the legend to toggle the legended line on and off
"""
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 0.2, 0.1)
y1 = 2*np.sin(2*np.pi*t)
y2 = 4*np.sin(2*np.pi*2*t)

fig, ax = plt.subplots()
ax.set_title('Click on legend line to toggle line on/off')
line1, = ax.plot(t, y1, lw=2, color='red', label='1 HZ')
line2, = ax.plot(t, y2, lw=2, color='blue', label='2 HZ')
leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
leg.get_frame().set_alpha(0.4)


# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line
lines = [line1, line2]
lined = dict()
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline


def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()




********************************************
./event_handling/pipong.py
#!/usr/bin/env python
# A matplotlib based game of Pong illustrating one way to write interactive
# animation which are easily ported to multiple backends
# pipong.py was written by Paul Ivanov <http://pirsquared.org>

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint

instructions = """
Player A:       Player B:
  'e'      up     'i'
  'd'     down    'k'

press 't' -- close these instructions
            (animation will be much faster)
press 'a' -- add a puck
press 'A' -- remove a puck
press '1' -- slow down all pucks
press '2' -- speed up all pucks
press '3' -- slow down distractors
press '4' -- speed up distractors
press ' ' -- reset the first puck
press 'n' -- toggle distractors on/off
press 'g' -- toggle the game on/off

  """

class Pad(object):
    def __init__(self, disp,x,y,type='l'):
        self.disp = disp
        self.x = x
        self.y = y
        self.w = .3
        self.score = 0
        self.xoffset = 0.3
        self.yoffset = 0.1
        if type=='r':
            self.xoffset *= -1.0

        if type=='l' or type=='r':
            self.signx = -1.0
            self.signy = 1.0
        else:
            self.signx = 1.0
            self.signy = -1.0
    def contains(self, loc):
        return self.disp.get_bbox().contains(loc.x,loc.y)

class Puck(object):
    def __init__(self, disp, pad, field):
        self.vmax= .2
        self.disp = disp
        self.field = field
        self._reset(pad)
    def _reset(self,pad):
        self.x = pad.x + pad.xoffset
        if pad.y < 0:
            self.y = pad.y +  pad.yoffset
        else:
            self.y = pad.y - pad.yoffset
        self.vx = pad.x - self.x
        self.vy = pad.y + pad.w/2 - self.y
        self._speedlimit()
        self._slower()
        self._slower()
    def update(self,pads):
        self.x += self.vx
        self.y += self.vy
        for pad in pads:
            if pad.contains(self):
                self.vx *= 1.2 *pad.signx
                self.vy *= 1.2 *pad.signy
        fudge = .001
        #probably cleaner with something like...if not self.field.contains(self.x, self.y):
        if self.x < 0+fudge:
            #print "player A loses"
            pads[1].score += 1;
            self._reset(pads[0])
            return True
        if self.x > 7-fudge:
            #print "player B loses"
            pads[0].score += 1;
            self._reset(pads[1])
            return True
        if self.y < -1+fudge or self.y > 1-fudge:
            self.vy *= -1.0
            # add some randomness, just to make it interesting
            self.vy -= (randn()/300.0 + 1/300.0) * np.sign(self.vy)
        self._speedlimit()
        return False
    def _slower(self):
        self.vx /= 5.0
        self.vy /= 5.0
    def _faster(self):
        self.vx *= 5.0
        self.vy *= 5.0
    def _speedlimit(self):
        if self.vx > self.vmax:
            self.vx = self.vmax
        if self.vx < -self.vmax:
            self.vx = -self.vmax

        if self.vy > self.vmax:
            self.vy = self.vmax
        if self.vy < -self.vmax:
            self.vy = -self.vmax

class Game(object):

    def __init__(self, ax):
        # create the initial line
        self.ax = ax
        padAx = padBx= .50
        padAy = padBy= .30
        padBx+=6.3
        pA, = self.ax.barh(padAy,.2, height=.3,color='k', alpha=.5, edgecolor='b',lw=2,label="Player B", animated=True)
        pB, = self.ax.barh(padBy,.2, height=.3, left=padBx, color='k',alpha=.5, edgecolor='r',lw=2,label="Player A",animated=True)

        # distractors
        self.x = np.arange(0,2.22*np.pi,0.01)
        self.line, = self.ax.plot(self.x, np.sin(self.x),"r", animated=True, lw=4)
        self.line2, = self.ax.plot(self.x, np.cos(self.x),"g", animated=True, lw=4)
        self.line3, = self.ax.plot(self.x, np.cos(self.x),"g", animated=True, lw=4)
        self.line4, = self.ax.plot(self.x, np.cos(self.x),"r", animated=True, lw=4)
        self.centerline,= self.ax.plot([3.5,3.5], [1,-1],'k',alpha=.5, animated=True,  lw=8)
        self.puckdisp = self.ax.scatter([1],[1],label='_nolegend_', s=200,c='g',alpha=.9,animated=True)

        self.canvas = self.ax.figure.canvas
        self.background = None
        self.cnt = 0
        self.distract = True
        self.res = 100.0
        self.on = False
        self.inst = True    # show instructions from the beginning
        self.background = None
        self.pads = []
        self.pads.append( Pad(pA,0,padAy))
        self.pads.append( Pad(pB,padBx,padBy,'r'))
        self.pucks =[]
        self.i = self.ax.annotate(instructions,(.5,0.5),
                     name='monospace',
                     verticalalignment='center',
                     horizontalalignment='center',
                     multialignment='left',
                     textcoords='axes fraction',animated=True )
        self.canvas.mpl_connect('key_press_event', self.key_press)

    def draw(self, evt):
        draw_artist = self.ax.draw_artist
        if self.background is None:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # restore the clean slate background
        self.canvas.restore_region(self.background)

        # show the distractors
        if self.distract:
            self.line.set_ydata(np.sin(self.x+self.cnt/self.res))
            self.line2.set_ydata(np.cos(self.x-self.cnt/self.res))
            self.line3.set_ydata(np.tan(self.x+self.cnt/self.res))
            self.line4.set_ydata(np.tan(self.x-self.cnt/self.res))
            draw_artist(self.line)
            draw_artist(self.line2)
            draw_artist(self.line3)
            draw_artist(self.line4)

        # show the instructions - this is very slow
        if self.inst:
            self.ax.draw_artist(self.i)

        # pucks and pads
        if self.on:
            self.ax.draw_artist(self.centerline)
            for pad in self.pads:
                pad.disp.set_y(pad.y)
                pad.disp.set_x(pad.x)
                self.ax.draw_artist(pad.disp)

            for puck in self.pucks:
                if puck.update(self.pads):
                    # we only get here if someone scored
                    self.pads[0].disp.set_label("   "+ str(self.pads[0].score))
                    self.pads[1].disp.set_label("   "+ str(self.pads[1].score))
                    self.ax.legend(loc='center')
                    self.leg = self.ax.get_legend()
                    #self.leg.draw_frame(False) #don't draw the legend border
                    self.leg.get_frame().set_alpha(.2)
                    plt.setp(self.leg.get_texts(),fontweight='bold',fontsize='xx-large')
                    self.leg.get_frame().set_facecolor('0.2')
                    self.background = None
                    self.ax.figure.canvas.draw()
                    return True
                puck.disp.set_offsets([puck.x,puck.y])
                self.ax.draw_artist(puck.disp)


        # just redraw the axes rectangle
        self.canvas.blit(self.ax.bbox)

        if self.cnt==50000:
            # just so we don't get carried away
            print("...and you've been playing for too long!!!")
            plt.close()

        self.cnt += 1
        return True

    def key_press(self,event):
        if event.key == '3':
            self.res *= 5.0
        if event.key == '4':
            self.res /= 5.0

        if event.key == 'e':
            self.pads[0].y += .1
            if self.pads[0].y > 1 - .3:
                self.pads[0].y = 1-.3
        if event.key == 'd':
            self.pads[0].y -= .1
            if self.pads[0].y < -1:
                self.pads[0].y = -1

        if event.key == 'i':
            self.pads[1].y += .1
            if self.pads[1].y > 1 - .3:
                self.pads[1].y = 1-.3
        if event.key == 'k':
            self.pads[1].y -= .1
            if self.pads[1].y < -1:
                self.pads[1].y = -1

        if event.key == 'a':
            self.pucks.append(Puck(self.puckdisp,self.pads[randint(2)],self.ax.bbox))
        if event.key == 'A' and len(self.pucks):
            self.pucks.pop()
        if event.key == ' ' and len(self.pucks):
            self.pucks[0]._reset(self.pads[randint(2)])
        if event.key == '1':
            for p in self.pucks:
                p._slower()
        if event.key == '2':
            for p in self.pucks:
                p._faster()

        if event.key == 'n':
            self.distract = not self.distract

        if event.key == 'g':
            #self.ax.clear()
            self.on = not self.on
        if event.key == 't':
            self.inst = not self.inst
            self.i.set_visible(self.i.get_visible())
        if event.key == 'q':
            plt.close()




********************************************
./event_handling/poly_editor.py
"""
This is an example to show how to build cross-GUI applications using
matplotlib event handling to interact with objects on the canvas

"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment


class PolygonInteractor:
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)
        #self._update_line(poly)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas


    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state


    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind]>=self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes==None: return
        if event.button != 1: return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes: return
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts: self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = [tup for i,tup in enumerate(self.poly.xy) if i!=ind]
                self.line.set_data(zip(*self.poly.xy))
        elif event.key=='i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y # display coords
            for i in range(len(xys)-1):
                s0 = xys[i]
                s1 = xys[i+1]
                d = dist_point_to_segment(p, s0, s1)
                if d<=self.epsilon:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i:]))
                    self.line.set_data(zip(*self.poly.xy))
                    break


        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x,y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    theta = np.arange(0, 2*np.pi, 0.1)
    r = 1.5

    xs = r*np.cos(theta)
    ys = r*np.sin(theta)

    poly = Polygon(list(zip(xs, ys)), animated=True)

    fig, ax = plt.subplots()
    ax.add_patch(poly)
    p = PolygonInteractor(ax, poly)

    #ax.add_line(p.line)
    ax.set_title('Click and drag a point to move it')
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    plt.show()





********************************************
./event_handling/keypress_demo.py
#!/usr/bin/env python

"""
Show how to connect to keypress events
"""
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key=='x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')

plt.show()




********************************************
./event_handling/lasso_demo.py
"""
Show how to use a lasso to select a set of points and get the indices
of the selected points.  A callback is used to change the color of the
selected points

This is currently a proof-of-concept implementation (though it is
usable as is).  There will be some refinement of the API.
"""
from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path

import matplotlib.pyplot as plt
from numpy import nonzero
from numpy.random import rand

class Datum(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('blue')
    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include: self.color = self.colorin
        else: self.color = self.colorout


class LassoManager(object):
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        fig = ax.figure
        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(100,),
            facecolors=facecolors,
            offsets = self.xys,
            transOffset = ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = Datum.colorin
            else:
                facecolors[i] = Datum.colorout

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

if __name__ == '__main__':

    data = [Datum(*xy) for xy in rand(100, 2)]

    ax = plt.axes(xlim=(0,1), ylim=(0,1), autoscale_on=False)
    lman = LassoManager(ax, data)

    plt.show()




********************************************
./event_handling/pick_event_demo.py
#!/usr/bin/env python

"""

You can enable picking by setting the "picker" property of an artist
(for example, a matplotlib Line2D, Text, Patch, Polygon, AxesImage,
etc...)

There are a variety of meanings of the picker property

    None -  picking is disabled for this artist (default)

    boolean - if True then picking will be enabled and the
      artist will fire a pick event if the mouse event is over
      the artist

    float - if picker is a number it is interpreted as an
      epsilon tolerance in points and the artist will fire
      off an event if it's data is within epsilon of the mouse
      event.  For some artists like lines and patch collections,
      the artist may provide additional data to the pick event
      that is generated, for example, the indices of the data within
      epsilon of the pick event

    function - if picker is callable, it is a user supplied
      function which determines whether the artist is hit by the
      mouse event.

         hit, props = picker(artist, mouseevent)

      to determine the hit test.  If the mouse event is over the
      artist, return hit=True and props is a dictionary of properties
      you want added to the PickEvent attributes


After you have enabled an artist for picking by setting the "picker"
property, you need to connect to the figure canvas pick_event to get
pick callbacks on mouse press events.  For example,

  def pick_handler(event):
      mouseevent = event.mouseevent
      artist = event.artist
      # now do something with this...


The pick event (matplotlib.backend_bases.PickEvent) which is passed to
your callback is always fired with two attributes:

  mouseevent - the mouse event that generate the pick event.  The
    mouse event in turn has attributes like x and y (the coordinates in
    display space, such as pixels from left, bottom) and xdata, ydata (the
    coords in data space).  Additionally, you can get information about
    which buttons were pressed, which keys were pressed, which Axes
    the mouse is over, etc.  See matplotlib.backend_bases.MouseEvent
    for details.

  artist - the matplotlib.artist that generated the pick event.

Additionally, certain artists like Line2D and PatchCollection may
attach additional meta data like the indices into the data that meet
the picker criteria (for example, all the points in the line that are within
the specified epsilon tolerance)

The examples below illustrate each of these methods.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

if 1: # simple picking, lines, rectangles and text
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title('click on points, rectangles or text', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance

    # pick the rectangle
    bars = ax2.bar(range(10), rand(10), picker=True)
    for label in ax2.get_xticklabels():  # make the xtick labels pickable
        label.set_picker(True)


    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick1 patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onpick1 text:', text.get_text())



    fig.canvas.mpl_connect('pick_event', onpick1)

if 1: # picking with a custom hit test function
    # you can define custom pickers by setting picker to a callable
    # function.  The function has the signature
    #
    #  hit, props = func(artist, mouseevent)
    #
    # to determine the hit test.  if the mouse event is over the artist,
    # return hit=True and props is a dictionary of
    # properties you want added to the PickEvent attributes

    def line_picker(line, mouseevent):
        """
        find the points within a certain distance from the mouseclick in
        data coords and attach some extra attributes, pickx and picky
        which are the data points that were picked
        """
        if mouseevent.xdata is None: return False, dict()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        maxd = 0.05
        d = np.sqrt((xdata-mouseevent.xdata)**2. + (ydata-mouseevent.ydata)**2.)

        ind = np.nonzero(np.less_equal(d, maxd))
        if len(ind):
            pickx = np.take(xdata, ind)
            picky = np.take(ydata, ind)
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()

    def onpick2(event):
        print('onpick2 line:', event.pickx, event.picky)

    fig, ax = plt.subplots()
    ax.set_title('custom picker for line data')
    line, = ax.plot(rand(100), rand(100), 'o', picker=line_picker)
    fig.canvas.mpl_connect('pick_event', onpick2)


if 1: # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    x, y, c, s = rand(4, 100)
    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))

    fig, ax = plt.subplots()
    col = ax.scatter(x, y, 100*s, c, picker=True)
    #fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', onpick3)

if 1: # picking images (matplotlib.image.AxesImage)
    fig, ax = plt.subplots()
    im1 = ax.imshow(rand(10,5), extent=(1,2,1,2), picker=True)
    im2 = ax.imshow(rand(5,10), extent=(3,4,1,2), picker=True)
    im3 = ax.imshow(rand(20,25), extent=(1,2,3,4), picker=True)
    im4 = ax.imshow(rand(30,12), extent=(3,4,3,4), picker=True)
    ax.axis([0,5,0,5])

    def onpick4(event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            im = artist
            A = im.get_array()
            print('onpick4 image', A.shape)

    fig.canvas.mpl_connect('pick_event', onpick4)


plt.show()





********************************************
./event_handling/zoom_window.py
"""
This example shows how to connect events in one window, for example, a mouse
press, to another figure window.

If you click on a point in the first window, the z and y limits of the
second will be adjusted so that the center of the zoom in the second
window will be the x,y coordinates of the clicked point.

Note the diameter of the circles in the scatter are defined in
points**2, so their size is independent of the zoom
"""
from matplotlib.pyplot import figure, show
import numpy
figsrc = figure()
figzoom = figure()

axsrc = figsrc.add_subplot(111, xlim=(0,1), ylim=(0,1), autoscale_on=False)
axzoom = figzoom.add_subplot(111, xlim=(0.45,0.55), ylim=(0.4,.6),
                                                    autoscale_on=False)
axsrc.set_title('Click to zoom')
axzoom.set_title('zoom window')
x,y,s,c = numpy.random.rand(4,200)
s *= 200


axsrc.scatter(x,y,s,c)
axzoom.scatter(x,y,s,c)

def onpress(event):
    if event.button!=1: return
    x,y = event.xdata, event.ydata
    axzoom.set_xlim(x-0.1, x+0.1)
    axzoom.set_ylim(y-0.1, y+0.1)
    figzoom.canvas.draw()

figsrc.canvas.mpl_connect('button_press_event', onpress)
show()





********************************************
./event_handling/resample.py
import numpy as np
import matplotlib.pyplot as plt
from scikits.audiolab import wavread

# A class that will downsample the data and recompute when zoomed.
class DataDisplayDownsampler(object):
    def __init__(self, xdata, ydata):
        self.origYData = ydata
        self.origXData = xdata
        self.numpts = 3000
        self.delta = xdata[-1] - xdata[0]

    def resample(self, xstart, xend):
        # Very simple downsampling that takes the points within the range
        # and picks every Nth point
        mask = (self.origXData > xstart) & (self.origXData < xend)
        xdata = self.origXData[mask]
        ratio = int(xdata.size / self.numpts) + 1
        xdata = xdata[::ratio]

        ydata = self.origYData[mask]
        ydata = ydata[::ratio]

        return xdata, ydata

    def update(self, ax):
        # Update the line
        lims = ax.viewLim
        if np.abs(lims.width - self.delta) > 1e-8:
            self.delta = lims.width
            xstart, xend = lims.intervalx
            self.line.set_data(*self.downsample(xstart, xend))
            ax.figure.canvas.draw_idle()

# Read data
data = wavread('/usr/share/sounds/purple/receive.wav')[0]
ydata = np.tile(data[:, 0], 100)
xdata = np.arange(ydata.size)

d = DataDisplayDownsampler(xdata, ydata)

fig, ax = plt.subplots()

#Hook up the line
xdata, ydata = d.downsample(xdata[0], xdata[-1])
d.line, = ax.plot(xdata, ydata)
ax.set_autoscale_on(False) # Otherwise, infinite loop

# Connect for changing the view limits
ax.callbacks.connect('xlim_changed', d.update)

plt.show()




********************************************
./event_handling/figure_axes_enter_leave.py
"""
Illustrate the figure and axes enter and leave events by changing the
frame colors on enter and leave
"""
from __future__ import print_function
import matplotlib.pyplot as plt

def enter_axes(event):
    print('enter_axes', event.inaxes)
    event.inaxes.patch.set_facecolor('yellow')
    event.canvas.draw()

def leave_axes(event):
    print('leave_axes', event.inaxes)
    event.inaxes.patch.set_facecolor('white')
    event.canvas.draw()

def enter_figure(event):
    print('enter_figure', event.canvas.figure)
    event.canvas.figure.patch.set_facecolor('red')
    event.canvas.draw()

def leave_figure(event):
    print('leave_figure', event.canvas.figure)
    event.canvas.figure.patch.set_facecolor('grey')
    event.canvas.draw()

fig1, (ax, ax2) = plt.subplots(2, 1)
fig1.suptitle('mouse hover over figure or axes to trigger events')

fig1.canvas.mpl_connect('figure_enter_event', enter_figure)
fig1.canvas.mpl_connect('figure_leave_event', leave_figure)
fig1.canvas.mpl_connect('axes_enter_event', enter_axes)
fig1.canvas.mpl_connect('axes_leave_event', leave_axes)

fig2, (ax, ax2) = plt.subplots(2, 1)
fig2.suptitle('mouse hover over figure or axes to trigger events')

fig2.canvas.mpl_connect('figure_enter_event', enter_figure)
fig2.canvas.mpl_connect('figure_leave_event', leave_figure)
fig2.canvas.mpl_connect('axes_enter_event', enter_axes)
fig2.canvas.mpl_connect('axes_leave_event', leave_axes)

plt.show()






********************************************
./event_handling/test_mouseclicks.py
#!/usr/bin/env python
from __future__ import print_function

import matplotlib
#matplotlib.use("WxAgg")
#matplotlib.use("TkAgg")
#matplotlib.use("GTKAgg")
#matplotlib.use("Qt4Agg")
#matplotlib.use("CocoaAgg")
#matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

#print("***** TESTING WITH BACKEND: %s"%matplotlib.get_backend() + " *****")


def OnClick(event):
    if event.dblclick:
        print("DBLCLICK", event)
    else:
        print("DOWN    ", event)


def OnRelease(event):
    print("UP      ", event)


fig = plt.gcf()
cid_up = fig.canvas.mpl_connect('button_press_event', OnClick)
cid_down = fig.canvas.mpl_connect('button_release_event', OnRelease)

plt.gca().text(0.5, 0.5, "Click on the canvas to test mouse events.",
               ha="center", va="center")

plt.show()




********************************************
./event_handling/viewlims.py
# Creates two identical panels.  Zooming in on the right panel will show
# a rectangle in the first panel, denoting the zoomed region.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# We just subclass Rectangle so that it can be called with an Axes
# instance, causing the rectangle to update its shape to match the
# bounds of the Axes
class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()

# A class that will regenerate a fractal set as we zoom in, so that you
# can actually see the increasing detail.  A box in the left panel will show
# the area to which we are zoomed.
class MandlebrotDisplay(object):
    def __init__(self, h=500, w=500, niter=50, radius=2., power=2):
        self.height = h
        self.width = w
        self.niter = niter
        self.radius = radius
        self.power = power

    def __call__(self, xstart, xend, ystart, yend):
        self.x = np.linspace(xstart, xend, self.width)
        self.y = np.linspace(ystart, yend, self.height).reshape(-1,1)
        c = self.x + 1.0j * self.y
        threshold_time = np.zeros((self.height, self.width))
        z = np.zeros(threshold_time.shape, dtype=np.complex)
        mask = np.ones(threshold_time.shape, dtype=np.bool)
        for i in range(self.niter):
            z[mask] = z[mask]**self.power + c[mask]
            mask = (np.abs(z) < self.radius)
            threshold_time += mask
        return threshold_time

    def ax_update(self, ax):
        ax.set_autoscale_on(False) # Otherwise, infinite loop

        #Get the number of points from the number of pixels in the window
        dims = ax.axesPatch.get_window_extent().bounds
        self.width = int(dims[2] + 0.5)
        self.height = int(dims[2] + 0.5)

        #Get the range for the new area
        xstart,ystart,xdelta,ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        yend = ystart + ydelta

        # Update the image object with our new data and extent
        im = ax.images[-1]
        im.set_data(self.__call__(xstart, xend, ystart, yend))
        im.set_extent((xstart, xend, ystart, yend))
        ax.figure.canvas.draw_idle()

md = MandlebrotDisplay()
Z = md(-2., 0.5, -1.25, 1.25)

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Z, origin='lower', extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))
ax2.imshow(Z, origin='lower', extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))

rect = UpdatingRect([0, 0], 0, 0, facecolor='None', edgecolor='black')
rect.set_bounds(*ax2.viewLim.bounds)
ax1.add_patch(rect)

# Connect for changing the view limits
ax2.callbacks.connect('xlim_changed', rect)
ax2.callbacks.connect('ylim_changed', rect)

ax2.callbacks.connect('xlim_changed', md.ax_update)
ax2.callbacks.connect('ylim_changed', md.ax_update)

plt.show()




********************************************
./event_handling/data_browser.py
import numpy as np


class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """
    def __init__(self):
        self.lastind = 0

        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.selected,  = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
                                  color='yellow', visible=False)

    def onpress(self, event):
        if self.lastind is None: return
        if event.key not in ('n', 'p'): return
        if event.key=='n': inc = 1
        else:  inc = -1


        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs)-1)
        self.update()

    def onpick(self, event):

       if event.artist!=line: return True

       N = len(event.ind)
       if not N: return True

       # the click locations
       x = event.mouseevent.xdata
       y = event.mouseevent.ydata


       distances = np.hypot(x-xs[event.ind], y-ys[event.ind])
       indmin = distances.argmin()
       dataind = event.ind[indmin]

       self.lastind = dataind
       self.update()

    def update(self):
        if self.lastind is None: return

        dataind = self.lastind

        ax2.cla()
        ax2.plot(X[dataind])

        ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),
                 transform=ax2.transAxes, va='top')
        ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        self.text.set_text('selected: %d'%dataind)
        fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(100, 200)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('click on point to plot time series')
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance

    browser = PointBrowser()

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()





********************************************
./event_handling/idle_and_timeout.py
from __future__ import print_function
"""
Demonstrate/test the idle and timeout API

This is only tested on gtk so far and is a prototype implementation
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

t = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2*np.pi*t)
y2 = np.cos(2*np.pi*t)
line1, = ax.plot(y1)
line2, = ax.plot(y2)

N = 100
def on_idle(event):
    on_idle.count +=1
    print('idle', on_idle.count)
    line1.set_ydata(np.sin(2*np.pi*t*(N-on_idle.count)/float(N)))
    event.canvas.draw()
    # test boolean return removal
    if on_idle.count==N:
        return False
    return True
on_idle.cid = None
on_idle.count = 0

fig.canvas.mpl_connect('idle_event', on_idle)

plt.show()






********************************************
./event_handling/looking_glass.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
x, y = np.random.rand(2, 200)

fig, ax = plt.subplots()
circ = patches.Circle( (0.5, 0.5), 0.25, alpha=0.8, fc='yellow')
ax.add_patch(circ)


ax.plot(x, y, alpha=0.2)
line, = ax.plot(x, y, alpha=1.0, clip_path=circ)

class EventHandler:
   def __init__(self):
       fig.canvas.mpl_connect('button_press_event', self.onpress)
       fig.canvas.mpl_connect('button_release_event', self.onrelease)
       fig.canvas.mpl_connect('motion_notify_event', self.onmove)
       self.x0, self.y0 = circ.center
       self.pressevent = None

   def onpress(self, event):
      if event.inaxes!=ax:
         return

      if not circ.contains(event)[0]:
         return

      self.pressevent = event

   def onrelease(self, event):
      self.pressevent = None
      self.x0, self.y0 = circ.center

   def onmove(self, event):
      if self.pressevent is None or event.inaxes!=self.pressevent.inaxes:
         return

      dx = event.xdata - self.pressevent.xdata
      dy = event.ydata - self.pressevent.ydata
      circ.center = self.x0 + dx, self.y0 + dy
      line.set_clip_path(circ)
      fig.canvas.draw()

handler = EventHandler()
plt.show()




********************************************
./widgets/span_selector.py
#!/usr/bin/env python
"""
The SpanSelector is a mouse widget to select a xmin/xmax range and plot the
detail view of the selected region in the lower axes
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(211, axisbg='#FFFFCC')

x = np.arange(0.0, 5.0, 0.01)
y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))

ax.plot(x, y, '-')
ax.set_ylim(-2,2)
ax.set_title('Press left mouse button and drag to test')

ax2 = fig.add_subplot(212, axisbg='#FFFFCC')
line2, = ax2.plot(x, y, '-')


def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x)-1, indmax)

    thisx = x[indmin:indmax]
    thisy = y[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw()

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red') )


plt.show()




********************************************
./widgets/multicursor.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(t, s1)


ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, s2)

multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1)
plt.show()




********************************************
./widgets/slider_demo.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)
l, = plt.plot(t,s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()




********************************************
./widgets/check_buttons.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False, lw=2)
l1, = ax.plot(t, s1, lw=2)
l2, = ax.plot(t, s2, lw=2)
plt.subplots_adjust(left=0.2)

rax = plt.axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))

def func(label):
    if label == '2 Hz': l0.set_visible(not l0.get_visible())
    elif label == '4 Hz': l1.set_visible(not l1.get_visible())
    elif label == '6 Hz': l2.set_visible(not l2.get_visible())
    plt.draw()
check.on_clicked(func)

plt.show()




********************************************
./widgets/lasso_selector_demo.py
from __future__ import print_function

import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()
    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(ax, pts)

    plt.draw()
    raw_input('Press any key to accept selected points')
    print("Selected points:")
    print(selector.xys[selector.ind])
    selector.disconnect()

    # Block end of script so you can check that the lasso is disconnected.
    raw_input('Press any key to quit')




********************************************
./widgets/cursor.py
#!/usr/bin/env python

from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, axisbg='#FFFFCC')

x, y = 4*(np.random.rand(2, 100)-.5)
ax.plot(x, y, 'o')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# set useblit = True on gtkagg for enhanced performance
cursor = Cursor(ax, useblit=True, color='red', linewidth=2 )

plt.show()




********************************************
./widgets/rectangle_selector.py
from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked wether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print ("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print (" The button you used were: %s %s" % (eclick.button, erelease.button))

def toggle_selector(event):
    print (' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print (' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print (' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


fig, current_ax = plt.subplots()                    # make a new plotingrange
N = 100000                                       # If N is large one can see
x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

plt.plot(x, +np.sin(.2*np.pi*x), lw=3.5, c='b', alpha=.7)  # plot something
plt.plot(x, +np.cos(.2*np.pi*x), lw=3.5, c='r', alpha=.5)
plt.plot(x, -np.sin(.2*np.pi*x), lw=3.5, c='g', alpha=.3)

print ("\n      click  -->  release")

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1,3], # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')
plt.connect('key_press_event', toggle_selector)
plt.show()




********************************************
./widgets/radio_buttons.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(8*np.pi*t)

fig, ax = plt.subplots()
l, = ax.plot(t, s0, lw=2, color='red')
plt.subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.05, 0.7, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('2 Hz', '4 Hz', '8 Hz'))
def hzfunc(label):
    hzdict = {'2 Hz':s0, '4 Hz':s1, '8 Hz':s2}
    ydata = hzdict[label]
    l.set_ydata(ydata)
    plt.draw()
radio.on_clicked(hzfunc)

rax = plt.axes([0.05, 0.4, 0.15, 0.15], axisbg=axcolor)
radio2 = RadioButtons(rax, ('red', 'blue', 'green'))
def colorfunc(label):
    l.set_color(label)
    plt.draw()
radio2.on_clicked(colorfunc)

rax = plt.axes([0.05, 0.1, 0.15, 0.15], axisbg=axcolor)
radio3 = RadioButtons(rax, ('-', '--', '-.', 'steps', ':'))
def stylefunc(label):
    l.set_linestyle(label)
    plt.draw()
radio3.on_clicked(stylefunc)

plt.show()




********************************************
./widgets/buttons.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()





********************************************
./widgets/menu.py
from __future__ import division, print_function
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image


class ItemProperties:
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha

        self.labelcolor_rgb = colors.colorConverter.to_rgb(labelcolor)
        self.bgcolor_rgb = colors.colorConverter.to_rgb(bgcolor)

class MenuItem(artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 5
    pady = 5
    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None):
        artist.Artist.__init__(self)

        self.set_figure(fig)
        self.labelstr = labelstr

        if props is None:
            props = ItemProperties()

        if hoverprops is None:
            hoverprops = ItemProperties()

        self.props = props
        self.hoverprops = hoverprops


        self.on_select = on_select

        x, self.depth = self.parser.to_mask(
            labelstr, fontsize=props.fontsize, dpi=fig.dpi)

        if props.fontsize!=hoverprops.fontsize:
            raise NotImplementedError(
                        'support for different font sizes not implemented')


        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]

        self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
        self.labelArray[:, :, -1] = x/255.

        self.label = image.FigureImage(fig, origin='upper')
        self.label.set_array(self.labelArray)

        # we'll update these later
        self.rect = patches.Rectangle((0,0), 1,1)

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return

        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        print(x, y, w, h)
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)

        self.label.ox = x+self.padx
        self.label.oy = y-self.depth+self.pady/2.

        self.rect._update_patch_transform()
        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        r, g, b = props.labelcolor_rgb
        self.labelArray[:, :, 0] = r
        self.labelArray[:, :, 1] = g
        self.labelArray[:, :, 2] = b
        self.label.set_array(self.labelArray)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b,junk = self.rect.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)


        self.hover = b
        return changed

class Menu:

    def __init__(self, fig, menuitems):
        self.figure = fig
        fig.suppressComposite = True

        self.menuitems = menuitems
        self.numitems = len(menuitems)

        maxw = max([item.labelwidth for item in menuitems])
        maxh = max([item.labelheight for item in menuitems])


        totalh = self.numitems*maxh + (self.numitems+1)*2*MenuItem.pady


        x0 = 100
        y0 = 400

        width = maxw + 2*MenuItem.padx
        height = maxh+MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0-maxh-MenuItem.pady


            item.set_extent(left, bottom, width, height)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady


        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            draw = item.set_hover(event)
            if draw:
                self.figure.canvas.draw()
                break


fig = plt.figure()
fig.subplots_adjust(left=0.3)
props = ItemProperties(labelcolor='black', bgcolor='yellow',
                       fontsize=15, alpha=0.2)
hoverprops = ItemProperties(labelcolor='white', bgcolor='blue',
                            fontsize=15, alpha=0.2)

menuitems = []
for label in ('open', 'close', 'save', 'save as', 'quit'):
    def on_select(item):
        print('you selected %s' % item.labelstr)
    item = MenuItem(fig, label, props=props, hoverprops=hoverprops,
                    on_select=on_select)
    menuitems.append(item)

menu = Menu(fig, menuitems)
plt.show()




********************************************
./user_interfaces/mathtext_wx.py
"""
Demonstrates how to convert mathtext to a wx.Bitmap for display in various
controls on wxPython.
"""

import matplotlib
matplotlib.use("WxAgg")
from numpy import arange, sin, pi, cos, log
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import wx

IS_GTK = 'wxGTK' in wx.PlatformInfo
IS_WIN = 'wxMSW' in wx.PlatformInfo
IS_MAC = 'wxMac' in wx.PlatformInfo

############################################################
# This is where the "magic" happens.
from matplotlib.mathtext import MathTextParser
mathtext_parser = MathTextParser("Bitmap")
def mathtext_to_wxbitmap(s):
    ftimage, depth = mathtext_parser.parse(s, 150)
    return wx.BitmapFromBufferRGBA(
        ftimage.get_width(), ftimage.get_height(),
        ftimage.as_rgba_str())
############################################################

functions = [
    (r'$\sin(2 \pi x)$'      , lambda x: sin(2*pi*x)),
    (r'$\frac{4}{3}\pi x^3$' , lambda x: (4.0 / 3.0) * pi * x**3),
    (r'$\cos(2 \pi x)$'      , lambda x: cos(2*pi*x)),
    (r'$\log(x)$'            , lambda x: log(x))
]

class CanvasFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, -1, title, size=(550, 350))
        self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.change_plot(0)

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.add_buttonbar()
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.add_toolbar()  # comment this out for no toolbar

        menuBar = wx.MenuBar()

        # File Menu
        menu = wx.Menu()
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")
        menuBar.Append(menu, "&File")

        if IS_GTK or IS_WIN:
            # Equation Menu
            menu = wx.Menu()
            for i, (mt, func) in enumerate(functions):
                bm = mathtext_to_wxbitmap(mt)
                item = wx.MenuItem(menu, 1000 + i, "")
                item.SetBitmap(bm)
                menu.AppendItem(item)
                self.Bind(wx.EVT_MENU, self.OnChangePlot, item)
            menuBar.Append(menu, "&Functions")

        self.SetMenuBar(menuBar)

        self.SetSizer(self.sizer)
        self.Fit()

    def add_buttonbar(self):
        self.button_bar = wx.Panel(self)
        self.button_bar_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.button_bar, 0, wx.LEFT | wx.TOP | wx.GROW)

        for i, (mt, func) in enumerate(functions):
            bm = mathtext_to_wxbitmap(mt)
            button = wx.BitmapButton(self.button_bar, 1000 + i, bm)
            self.button_bar_sizer.Add(button, 1, wx.GROW)
            self.Bind(wx.EVT_BUTTON, self.OnChangePlot, button)

        self.button_bar.SetSizer(self.button_bar_sizer)

    def add_toolbar(self):
        """Copied verbatim from embedding_wx2.py"""
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if IS_MAC:
            self.SetToolBar(self.toolbar)
        else:
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.update()

    def OnPaint(self, event):
        self.canvas.draw()

    def OnChangePlot(self, event):
        self.change_plot(event.GetId() - 1000)

    def change_plot(self, plot_number):
        t = arange(1.0,3.0,0.01)
        s = functions[plot_number][1](t)
        self.axes.clear()
        self.axes.plot(t, s)
        self.Refresh()

class MyApp(wx.App):
    def OnInit(self):
        frame = CanvasFrame(None, "wxPython mathtext demo app")
        self.SetTopWindow(frame)
        frame.Show(True)
        return True

app = MyApp()
app.MainLoop()





********************************************
./user_interfaces/pylab_with_gtk.py
"""
An example of how to use pylab to manage your figure windows, but
modify the GUI by accessing the underlying gtk widgets
"""
from __future__ import print_function
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
plt.plot([1,2,3], 'ro-', label='easy as 1 2 3')
plt.plot([1,4,9], 'gs--', label='easy as 1 2 3 squared')
plt.legend()


manager = plt.get_current_fig_manager()
# you can also access the window or vbox attributes this way
toolbar = manager.toolbar

# now let's add a button to the toolbar
import gtk
next = 8; #where to insert this in the mpl toolbar
button = gtk.Button('Click me')
button.show()

def clicked(button):
    print('hi mom')
button.connect('clicked', clicked)

toolitem = gtk.ToolItem()
toolitem.show()
toolitem.set_tooltip(
    toolbar.tooltips,
    'Click me for fun and profit')

toolitem.add(button)
toolbar.insert(toolitem, next); next +=1

# now let's add a widget to the vbox
label = gtk.Label()
label.set_markup('Drag mouse over axes for position')
label.show()
vbox = manager.vbox
vbox.pack_start(label, False, False)
vbox.reorder_child(manager.toolbar, -1)

def update(event):
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup('<span color="#ef0000">x,y=(%f, %f)</span>'%(event.xdata, event.ydata))

plt.connect('motion_notify_event', update)

plt.show()




********************************************
./user_interfaces/embedding_in_gtk3_panzoom.py
#!/usr/bin/env python
"""
demonstrate NavigationToolbar with GTK3 accessed via pygobject
"""

from gi.repository import Gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit )
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(1,1,1)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
a.plot(t,s)

vbox = Gtk.VBox()
win.add(vbox)

# Add canvas to vbox
canvas = FigureCanvas(f)  # a Gtk.DrawingArea
vbox.pack_start(canvas, True, True, 0)

# Create toolbar
toolbar = NavigationToolbar(canvas, win)
vbox.pack_start(toolbar, False, False, 0)

win.show_all()
Gtk.main()





********************************************
./user_interfaces/gtk_spreadsheet.py
#!/usr/bin/env python
"""
Example of embedding matplotlib in an application and interacting with
a treeview to store data.  Double click on an entry to update plot
data

"""
import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

from numpy.random import random
from matplotlib.figure import Figure


class DataManager(gtk.Window):
    numRows, numCols = 20,10

    data = random((numRows, numCols))

    def __init__(self):
        gtk.Window.__init__(self)
        self.set_default_size(600, 600)
        self.connect('destroy', lambda win: gtk.main_quit())

        self.set_title('GtkListStore demo')
        self.set_border_width(8)

        vbox = gtk.VBox(False, 8)
        self.add(vbox)

        label = gtk.Label('Double click a row to plot the data')

        vbox.pack_start(label, False, False)

        sw = gtk.ScrolledWindow()
        sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        sw.set_policy(gtk.POLICY_NEVER,
                      gtk.POLICY_AUTOMATIC)
        vbox.pack_start(sw, True, True)

        model = self.create_model()

        self.treeview = gtk.TreeView(model)
        self.treeview.set_rules_hint(True)


        # matplotlib stuff
        fig = Figure(figsize=(6,4))

        self.canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        vbox.pack_start(self.canvas, True, True)
        ax = fig.add_subplot(111)
        self.line, = ax.plot(self.data[0,:], 'go')  # plot the first row

        self.treeview.connect('row-activated', self.plot_row)
        sw.add(self.treeview)

        self.add_columns()

        self.add_events(gdk.BUTTON_PRESS_MASK |
                        gdk.KEY_PRESS_MASK|
                        gdk.KEY_RELEASE_MASK)


    def plot_row(self, treeview, path, view_column):
        ind, = path  # get the index into data
        points = self.data[ind,:]
        self.line.set_ydata(points)
        self.canvas.draw()


    def add_columns(self):
        for i in range(self.numCols):
            column = gtk.TreeViewColumn('%d'%i, gtk.CellRendererText(), text=i)
            self.treeview.append_column(column)


    def create_model(self):
        types = [float]*self.numCols
        store = gtk.ListStore(*types)

        for row in self.data:
            store.append(row)
        return store


manager = DataManager()
manager.show_all()
gtk.main()




********************************************
./user_interfaces/embedding_in_wx4.py
#!/usr/bin/env python
"""
An example of how to use wx or wxagg in an application with a custom
toolbar
"""

# Used to guarantee to use at least Wx2.8
import wxversion
wxversion.ensureMinimal('2.8')

from numpy import arange, sin, pi

import matplotlib

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg

from matplotlib.backends.backend_wx import _load_bitmap
from matplotlib.figure import Figure
from numpy.random import rand

import wx

class MyNavigationToolbar(NavigationToolbar2WxAgg):
    """
    Extend the default wx toolbar with your own event handlers
    """
    ON_CUSTOM = wx.NewId()
    def __init__(self, canvas, cankill):
        NavigationToolbar2WxAgg.__init__(self, canvas)

        # for simplicity I'm going to reuse a bitmap from wx, you'll
        # probably want to add your own.
        self.AddSimpleTool(self.ON_CUSTOM, _load_bitmap('stock_left.xpm'),
                           'Click me', 'Activate custom contol')
        wx.EVT_TOOL(self, self.ON_CUSTOM, self._on_custom)

    def _on_custom(self, evt):
        # add some text to the axes in a random location in axes (0,1)
        # coords) with a random color

        # get the axes
        ax = self.canvas.figure.axes[0]

        # generate a random location can color
        x,y = tuple(rand(2))
        rgb = tuple(rand(3))

        # add the text and draw
        ax.text(x, y, 'You clicked me',
                transform=ax.transAxes,
                color=rgb)
        self.canvas.draw()
        evt.Skip()


class CanvasFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure(figsize=(5,4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)

        self.axes.plot(t,s)

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)
        # Capture the paint message
        wx.EVT_PAINT(self, self.OnPaint)

        self.toolbar = MyNavigationToolbar(self.canvas, True)
        self.toolbar.Realize()
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)

        # update the axes menu on the toolbar
        self.toolbar.update()
        self.SetSizer(self.sizer)
        self.Fit()


    def OnPaint(self, event):
        self.canvas.draw()
        event.Skip()

class App(wx.App):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(True)

        return True

app = App(0)
app.MainLoop()




********************************************
./user_interfaces/embedding_in_qt4_wtoolbar.py
from __future__ import print_function

import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QTAgg as NavigationToolbar)
from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE

if use_pyside:
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        #self.x, self.y = self.get_data()
        self.data = self.get_data2()
        self.create_main_frame()
        self.on_draw()

    def create_main_frame(self):
        self.main_frame = QWidget()

        self.fig = Figure((5.0, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
        vbox.addWidget(self.mpl_toolbar)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def get_data2(self):
        return np.arange(20).reshape([4, 5]).copy()

    def on_draw(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        #self.axes.plot(self.x, self.y, 'ro')
        self.axes.imshow(self.data, interpolation='nearest')
        #self.axes.plot([1,2,3])
        self.canvas.draw()

    def on_key_press(self, event):
        print('you pressed', event.key)
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        key_press_handler(event, self.canvas, self.mpl_toolbar)


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()




********************************************
./user_interfaces/embedding_in_gtk3.py
#!/usr/bin/env python
"""
demonstrate adding a FigureCanvasGTK3Agg widget to a Gtk.ScrolledWindow
using GTK3 accessed via pygobject
"""

from gi.repository import Gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit )
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
a.plot(t,s)

sw = Gtk.ScrolledWindow()
win.add (sw)
# A scrolled window border goes outside the scrollbars and viewport
sw.set_border_width (10)

canvas = FigureCanvas(f)  # a Gtk.DrawingArea
canvas.set_size_request(800,600)
sw.add_with_viewport (canvas)

win.show_all()
Gtk.main()





********************************************
./user_interfaces/fourier_demo_wx.py
import numpy as np
import wx

import matplotlib
matplotlib.interactive(False)
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf, setp


class Knob:
    """
    Knob - simple class with a "setKnob" method.  
    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    Base class is for documentation purposes.
    """
    def setKnob(self, value):
        pass


class Param:
    """
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.  
    The class allows a cleaner way to update/"feedback" to the other knobs when 
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """
    def __init__(self, initialValue=None, minimum=0., maximum=1.):
        self.minimum = minimum
        self.maximum = maximum
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []
        
    def attach(self, knob):
        self.knobs += [knob]
        
    def set(self, value, knob=None):
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value


class SliderGroup(Knob):
    def __init__(self, parent, label, param):
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)
        self.slider = wx.Slider(parent, -1)
        self.slider.SetMax(param.maximum*1000)
        self.setKnob(param.value)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=2)
        sizer.Add(self.sliderText, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        self.param = param
        self.param.attach(self)

    def sliderHandler(self, evt):
        value = evt.GetInt() / 1000.
        self.param.set(value)
        
    def sliderTextHandler(self, evt):
        value = float(self.sliderText.GetValue())
        self.param.set(value)
        
    def setKnob(self, value):
        self.sliderText.SetValue('%g'%value)
        self.slider.SetValue(value*1000)


class FourierDemoFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.fourierDemoWindow = FourierDemoWindow(self)
        self.frequencySliderGroup = SliderGroup(self, label='Frequency f0:', \
            param=self.fourierDemoWindow.f0)
        self.amplitudeSliderGroup = SliderGroup(self, label=' Amplitude a:', \
            param=self.fourierDemoWindow.A)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.fourierDemoWindow, 1, wx.EXPAND)
        sizer.Add(self.frequencySliderGroup.sizer, 0, \
            wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer.Add(self.amplitudeSliderGroup.sizer, 0, \
            wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        self.SetSizer(sizer)
        

class FourierDemoWindow(wx.Window, Knob):
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)
        self.lines = []
        self.figure = Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.callbacks.connect('button_press_event', self.mouseDown)
        self.canvas.callbacks.connect('motion_notify_event', self.mouseMotion)
        self.canvas.callbacks.connect('button_release_event', self.mouseUp)
        self.state = ''
        self.mouseInfo = (None, None, None, None)
        self.f0 = Param(2., minimum=0., maximum=6.)
        self.A = Param(1., minimum=0.01, maximum=2.)
        self.draw()
        
        # Not sure I like having two params attached to the same Knob,
        # but that is what we have here... it works but feels kludgy -
        # although maybe it's not too bad since the knob changes both params
        # at the same time (both f0 and A are affected during a drag)
        self.f0.attach(self)
        self.A.attach(self)
        self.Bind(wx.EVT_SIZE, self.sizeHandler)
       
    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())
        
    def mouseDown(self, evt):
        if self.lines[0] in self.figure.hitlist(evt):
            self.state = 'frequency'
        elif self.lines[1] in self.figure.hitlist(evt):
            self.state = 'time'
        else:
            self.state = ''
        self.mouseInfo = (evt.xdata, evt.ydata, max(self.f0.value, .1), self.A.value)

    def mouseMotion(self, evt):
        if self.state == '':
            return
        x, y = evt.xdata, evt.ydata
        if x is None:  # outside the axes
            return
        x0, y0, f0Init, AInit = self.mouseInfo
        self.A.set(AInit+(AInit*(y-y0)/y0), self)
        if self.state == 'frequency':
            self.f0.set(f0Init+(f0Init*(x-x0)/x0))
        elif self.state == 'time':
            if (x-x0)/x0 != -1.:
                self.f0.set(1./(1./f0Init+(1./f0Init*(x-x0)/x0)))
                    
    def mouseUp(self, evt):
        self.state = ''

    def draw(self):
        if not hasattr(self, 'subplot1'):
            self.subplot1 = self.figure.add_subplot(211)
            self.subplot2 = self.figure.add_subplot(212)
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        color = (1., 0., 0.)
        self.lines += self.subplot1.plot(x1, y1, color=color, linewidth=2)
        self.lines += self.subplot2.plot(x2, y2, color=color, linewidth=2)
        #Set some plot attributes
        self.subplot1.set_title("Click and drag waveforms to change frequency and amplitude", fontsize=12)
        self.subplot1.set_ylabel("Frequency Domain Waveform X(f)", fontsize = 8)
        self.subplot1.set_xlabel("frequency f", fontsize = 8)
        self.subplot2.set_ylabel("Time Domain Waveform x(t)", fontsize = 8)
        self.subplot2.set_xlabel("time t", fontsize = 8)
        self.subplot1.set_xlim([-6, 6])
        self.subplot1.set_ylim([0, 1])
        self.subplot2.set_xlim([-2, 2])
        self.subplot2.set_ylim([-2, 2])
        self.subplot1.text(0.05, .95, r'$X(f) = \mathcal{F}\{x(t)\}$', \
            verticalalignment='top', transform = self.subplot1.transAxes)
        self.subplot2.text(0.05, .95, r'$x(t) = a \cdot \cos(2\pi f_0 t) e^{-\pi t^2}$', \
            verticalalignment='top', transform = self.subplot2.transAxes)

    def compute(self, f0, A):
        f = np.arange(-6., 6., 0.02)
        t = np.arange(-2., 2., 0.01)
        x = A*np.cos(2*np.pi*f0*t)*np.exp(-np.pi*t**2)
        X = A/2*(np.exp(-np.pi*(f-f0)**2) + np.exp(-np.pi*(f+f0)**2))
        return f, X, t, x

    def repaint(self):
        self.canvas.draw()

    def setKnob(self, value):
        # Note, we ignore value arg here and just go by state of the params
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        setp(self.lines[0], xdata=x1, ydata=y1)
        setp(self.lines[1], xdata=x2, ydata=y2)
        self.repaint()


class App(wx.App):
    def OnInit(self):
        self.frame1 = FourierDemoFrame(parent=None, title="Fourier Demo", size=(640, 480))
        self.frame1.Show()
        return True
        
app = App()
app.MainLoop()




********************************************
./user_interfaces/rec_edit_gtk_custom.py
"""
generate an editable gtk treeview widget for record arrays with custom
formatting of the cells and show how to limit string entries to a list
of strings
"""
from __future__ import print_function
import gtk
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import mpl_toolkits.gtktools as gtktools


datafile = cbook.get_sample_data('demodata.csv', asfileobj=False)
r = mlab.csv2rec(datafile, converterd={'weekdays':str})


formatd = mlab.get_formatd(r)
formatd['date'] = mlab.FormatDate('%Y-%m-%d')
formatd['prices'] = mlab.FormatMillions(precision=1)
formatd['gain'] = mlab.FormatPercent(precision=2)

# use a drop down combo for weekdays
stringd = dict(weekdays=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
constant = ['clientid']   # block editing of this field


liststore = gtktools.RecListStore(r, formatd=formatd, stringd=stringd)
treeview = gtktools.RecTreeView(liststore, constant=constant)

def mycallback(liststore, rownum, colname, oldval, newval):
    print('verify: old=%s, new=%s, rec=%s'%(oldval, newval, liststore.r[rownum][colname]))

liststore.callbacks.connect('cell_changed', mycallback)

win = gtk.Window()
win.set_title('click to edit')
win.add(treeview)
win.show_all()
win.connect('delete-event', lambda *args: gtk.main_quit())
gtk.main()




********************************************
./user_interfaces/embedding_in_tk2.py
#!/usr/bin/env python
import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

def destroy(e): sys.exit()

root = Tk.Tk()
root.wm_title("Embedding in TK")
#root.bind("<Destroy>", destroy)


f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

a.plot(t,s)
a.set_title('Tk embedding')
a.set_xlabel('X axis label')
a.set_ylabel('Y label')


# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#toolbar = NavigationToolbar2TkAgg( canvas, root )
#toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

button = Tk.Button(master=root, text='Quit', command=sys.exit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()




********************************************
./user_interfaces/embedding_in_gtk.py
#!/usr/bin/env python
"""
show how to add a matplotlib FigureCanvasGTK or FigureCanvasGTKAgg widget to a
gtk.Window
"""

import gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi

# uncomment to select /GTK/GTKAgg/GTKCairo
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas


win = gtk.Window()
win.connect("destroy", lambda x: gtk.main_quit())
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
a.plot(t,s)

canvas = FigureCanvas(f)  # a gtk.DrawingArea
win.add(canvas)

win.show_all()
gtk.main()




********************************************
./user_interfaces/rec_edit_gtk_simple.py
"""
Load a CSV file into a record array and edit it in a gtk treeview
"""

import gtk
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import mpl_toolkits.gtktools as gtktools

datafile = cbook.get_sample_data('demodata.csv', asfileobj=False)
r = mlab.csv2rec(datafile, converterd={'weekdays':str})

liststore, treeview, win = gtktools.edit_recarray(r)
win.set_title('click to edit')
win.connect('delete-event', lambda *args: gtk.main_quit())
gtk.main()




********************************************
./user_interfaces/histogram_demo_canvasagg.py
#!/usr/bin/env python
"""
This is an example that shows you how to work directly with the agg
figure canvas to create a figure using the pythonic API.

In this example, the contents of the agg canvas are extracted to a
string, which can in turn be passed off to PIL or put in a numeric
array


"""
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.mlab import normpdf
from numpy.random import randn
import numpy

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)

canvas = FigureCanvasAgg(fig)

mu, sigma = 100, 15
x = mu + sigma*randn(10000)

# the histogram of the data
n, bins, patches = ax.hist(x, 50, normed=1)

# add a 'best fit' line
y = normpdf( bins, mu, sigma)
line, = ax.plot(bins, y, 'r--')
line.set_linewidth(1)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
ax.set_title(r'$\mathrm{Histogram of IQ: }\mu=100, \sigma=15$')

ax.set_xlim( (40, 160))
ax.set_ylim( (0, 0.03))

canvas.draw()

s = canvas.tostring_rgb()  # save this and convert to bitmap as needed

# get the figure dimensions for creating bitmaps or numpy arrays,
# etc.
l,b,w,h = fig.bbox.bounds
w, h = int(w), int(h)

if 0:
    # convert to a numpy array
    X = numpy.fromstring(s, numpy.uint8)
    X.shape = h, w, 3

if 0:
    # pass off to PIL
    from PIL import Image
    im = Image.fromstring( "RGB", (w,h), s)
    im.show()







********************************************
./user_interfaces/interactive2.py
#!/usr/bin/env python

from __future__ import print_function

#  GTK Interactive Console
#  (C) 2003, Jon Anderson
#  See www.python.org/2.2/license.html for
#  license details.
#
import gtk
import gtk.gdk

import code
import os, sys
import pango

import __builtin__
import __main__


banner = """GTK Interactive Python Console
Thanks to Jon Anderson
%s
""" % sys.version

banner += """

Welcome to matplotlib.

    help(matplotlib) -- some general information about matplotlib
    help(plotting) -- shows a list of plot specific commands

"""
class Completer:
  """
  Taken from rlcompleter, with readline references stripped, and a local dictionary to use.
  """
  def __init__(self, locals):
    self.locals = locals

  def complete(self, text, state):
    """Return the next possible completion for 'text'.
    This is called successively with state == 0, 1, 2, ... until it
    returns None.  The completion should begin with 'text'.

    """
    if state == 0:
      if "." in text:
        self.matches = self.attr_matches(text)
      else:
        self.matches = self.global_matches(text)
    try:
      return self.matches[state]
    except IndexError:
      return None

  def global_matches(self, text):
    """Compute matches when text is a simple name.

    Return a list of all keywords, built-in functions and names
    currently defines in __main__ that match.

    """
    import keyword
    matches = []
    n = len(text)
    for list in [keyword.kwlist,__builtin__.__dict__.keys(),__main__.__dict__.keys(), self.locals.keys()]:
      for word in list:
        if word[:n] == text and word != "__builtins__":
          matches.append(word)
    return matches

  def attr_matches(self, text):
    """Compute matches when text contains a dot.

    Assuming the text is of the form NAME.NAME....[NAME], and is
    evaluatable in the globals of __main__, it will be evaluated
    and its attributes (as revealed by dir()) are used as possible
    completions.  (For class instances, class members are are also
    considered.)

    WARNING: this can still invoke arbitrary C code, if an object
    with a __getattr__ hook is evaluated.

    """
    import re
    m = re.match(r"(\w+(\.\w+)*)\.(\w*)", text)
    if not m:
      return
    expr, attr = m.group(1, 3)
    object = eval(expr, __main__.__dict__, self.locals)
    words = dir(object)
    if hasattr(object,'__class__'):
      words.append('__class__')
      words = words + get_class_members(object.__class__)
    matches = []
    n = len(attr)
    for word in words:
      if word[:n] == attr and word != "__builtins__":
        matches.append("%s.%s" % (expr, word))
    return matches

def get_class_members(klass):
  ret = dir(klass)
  if hasattr(klass,'__bases__'):
     for base in klass.__bases__:
       ret = ret + get_class_members(base)
  return ret



class OutputStream:
  """
  A Multiplexing output stream.
  It can replace another stream, and tee output to the original stream and too
  a GTK textview.
  """
  def __init__(self,view,old_out,style):
    self.view = view
    self.buffer = view.get_buffer()
    self.mark = self.buffer.create_mark("End",self.buffer.get_end_iter(), False )
    self.out = old_out
    self.style = style
    self.tee = 1

  def write(self,text):
    if self.tee:
      self.out.write(text)

    end = self.buffer.get_end_iter()

    if not self.view  == None:
      self.view.scroll_to_mark(self.mark, 0, True, 1, 1)

    self.buffer.insert_with_tags(end,text,self.style)

class GTKInterpreterConsole(gtk.ScrolledWindow):
  """
  An InteractiveConsole for GTK. It's an actual widget,
  so it can be dropped in just about anywhere.
  """
  def __init__(self):
    gtk.ScrolledWindow.__init__(self)
    self.set_policy (gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)

    self.text = gtk.TextView()
    self.text.set_wrap_mode(True)

    self.interpreter = code.InteractiveInterpreter()

    self.completer = Completer(self.interpreter.locals)
    self.buffer = []
    self.history = []
    self.banner = banner
    self.ps1 = ">>> "
    self.ps2 = "... "

    self.text.add_events( gtk.gdk.KEY_PRESS_MASK )
    self.text.connect( "key_press_event", self.key_pressed )

    self.current_history = -1

    self.mark = self.text.get_buffer().create_mark("End",self.text.get_buffer().get_end_iter(), False )

            #setup colors
    self.style_banner = gtk.TextTag("banner")
    self.style_banner.set_property( "foreground", "saddle brown" )

    self.style_ps1 = gtk.TextTag("ps1")
    self.style_ps1.set_property( "foreground", "DarkOrchid4" )
    self.style_ps1.set_property( "editable", False )
    self.style_ps1.set_property("font", "courier" )

    self.style_ps2 = gtk.TextTag("ps2")
    self.style_ps2.set_property( "foreground", "DarkOliveGreen" )
    self.style_ps2.set_property( "editable", False  )
    self.style_ps2.set_property("font", "courier" )

    self.style_out = gtk.TextTag("stdout")
    self.style_out.set_property( "foreground", "midnight blue" )
    self.style_err = gtk.TextTag("stderr")
    self.style_err.set_property( "style", pango.STYLE_ITALIC )
    self.style_err.set_property( "foreground", "red" )

    self.text.get_buffer().get_tag_table().add(self.style_banner)
    self.text.get_buffer().get_tag_table().add(self.style_ps1)
    self.text.get_buffer().get_tag_table().add(self.style_ps2)
    self.text.get_buffer().get_tag_table().add(self.style_out)
    self.text.get_buffer().get_tag_table().add(self.style_err)

    self.stdout = OutputStream(self.text,sys.stdout,self.style_out)
    self.stderr = OutputStream(self.text,sys.stderr,self.style_err)

    sys.stderr = self.stderr
    sys.stdout = self.stdout

    self.current_prompt = None

    self.write_line(self.banner, self.style_banner)
    self.prompt_ps1()

    self.add(self.text)
    self.text.show()


  def reset_history(self):
    self.history = []

  def reset_buffer(self):
    self.buffer = []

  def prompt_ps1(self):
    self.current_prompt = self.prompt_ps1
    self.write_line(self.ps1,self.style_ps1)

  def prompt_ps2(self):
    self.current_prompt = self.prompt_ps2
    self.write_line(self.ps2,self.style_ps2)

  def write_line(self,text,style=None):
    start,end = self.text.get_buffer().get_bounds()
    if style==None:
      self.text.get_buffer().insert(end,text)
    else:
      self.text.get_buffer().insert_with_tags(end,text,style)

    self.text.scroll_to_mark(self.mark, 0, True, 1, 1)

  def push(self, line):

    self.buffer.append(line)
    if len(line) > 0:
      self.history.append(line)

    source = "\n".join(self.buffer)

    more = self.interpreter.runsource(source, "<<console>>")

    if not more:
      self.reset_buffer()

    return more

  def key_pressed(self,widget,event):
    if event.keyval == gtk.gdk.keyval_from_name('Return'):
      return self.execute_line()

    if event.keyval == gtk.gdk.keyval_from_name('Up'):
      self.current_history = self.current_history - 1
      if self.current_history < - len(self.history):
        self.current_history = - len(self.history)
      return self.show_history()
    elif event.keyval == gtk.gdk.keyval_from_name('Down'):
      self.current_history = self.current_history + 1
      if self.current_history > 0:
        self.current_history = 0
      return self.show_history()
    elif event.keyval == gtk.gdk.keyval_from_name( 'Home'):
      l = self.text.get_buffer().get_line_count() - 1
      start = self.text.get_buffer().get_iter_at_line_offset(l,4)
      self.text.get_buffer().place_cursor(start)
      return True
    elif event.keyval == gtk.gdk.keyval_from_name( 'space') and event.state & gtk.gdk.CONTROL_MASK:
      return self.complete_line()
    return False

  def show_history(self):
    if self.current_history == 0:
      return True
    else:
      self.replace_line( self.history[self.current_history] )
      return True

  def current_line(self):
    start,end = self.current_line_bounds()
    return self.text.get_buffer().get_text(start,end, True)

  def current_line_bounds(self):
    txt_buffer = self.text.get_buffer()
    l = txt_buffer.get_line_count() - 1

    start = txt_buffer.get_iter_at_line(l)
    if start.get_chars_in_line() >= 4:
      start.forward_chars(4)
    end = txt_buffer.get_end_iter()
    return start,end

  def replace_line(self,txt):
    start,end = self.current_line_bounds()
    self.text.get_buffer().delete(start,end)
    self.write_line(txt)

  def execute_line(self, line=None):
    if line is None:
      line = self.current_line()
      self.write_line("\n")
    else:
      self.write_line(line + "\n")


    more = self.push(line)

    self.text.get_buffer().place_cursor(self.text.get_buffer().get_end_iter())

    if more:
        self.prompt_ps2()
    else:
        self.prompt_ps1()


    self.current_history = 0

    self.window.raise_()

    return True

  def complete_line(self):
    line = self.current_line()
    tokens = line.split()
    token = tokens[-1]

    completions = []
    p = self.completer.complete(token,len(completions))
    while p != None:
      completions.append(p)
      p = self.completer.complete(token, len(completions))

    if len(completions) != 1:
      self.write_line("\n")
      self.write_line("\n".join(completions), self.style_ps1)
      self.write_line("\n")
      self.current_prompt()
      self.write_line(line)
    else:
      i = line.rfind(token)
      line = line[0:i] + completions[0]
      self.replace_line(line)

    return True


def main():
  w = gtk.Window()
  console = GTKInterpreterConsole()
  console.set_size_request(640,480)
  w.add(console)

  def destroy(arg=None):
      gtk.main_quit()

  def key_event(widget,event):
      if gtk.gdk.keyval_name( event.keyval) == 'd' and \
             event.state & gtk.gdk.CONTROL_MASK:
          destroy()
      return False

  w.connect("destroy", destroy)

  w.add_events( gtk.gdk.KEY_PRESS_MASK )
  w.connect( 'key_press_event', key_event)
  w.show_all()

  console.execute_line('import matplotlib')
  console.execute_line("matplotlib.use('GTKAgg')")
  console.execute_line('matplotlib.interactive(1)')
  console.execute_line('from pylab import *')


  if len(sys.argv)>1:
    fname = sys.argv[1]
    if not os.path.exists(fname):
      print('%s does not exist' % fname)
    for line in file(fname):
      line = line.strip()

      console.execute_line(line)
  gtk.main()

if __name__ == '__main__':
  main()




********************************************
./user_interfaces/embedding_in_wx2.py
#!/usr/bin/env python
"""
An example of how to use wx or wxagg in an application with the new
toolbar - comment out the setA_toolbar line for no toolbar
"""

# Used to guarantee to use at least Wx2.8
import wxversion
wxversion.ensureMinimal('2.8')

from numpy import arange, sin, pi

import matplotlib

# uncomment the following to use wx rather than wxagg
#matplotlib.use('WX')
#from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.figure import Figure

import wx

class CanvasFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)

        self.axes.plot(t,s)
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.add_toolbar()  # comment this out for no toolbar


    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()


    def OnPaint(self, event):
        self.canvas.draw()

class App(wx.App):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(True)

        return True

app = App(0)
app.MainLoop()




********************************************
./user_interfaces/embedding_webagg.py
"""
This example demonstrates how to embed matplotlib WebAgg interactive
plotting in your own web application and framework.  It is not
necessary to do all this if you merely want to display a plot in a
browser or use matplotlib's built-in Tornado-based server "on the
side".

The framework being used must support web sockets.
"""

import io

try:
    import tornado
except ImportError:
    raise RuntimeError("This example requires tornado.")
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.websocket


from matplotlib.backends.backend_webagg_core import (
    FigureManagerWebAgg, new_figure_manager_given_figure)
from matplotlib.figure import Figure

import numpy as np

import json


def create_figure():
    """
    Creates a simple example figure.
    """
    fig = Figure()
    a = fig.add_subplot(111)
    t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2 * np.pi * t)
    a.plot(t, s)
    return fig


# The following is the content of the web page.  You would normally
# generate this using some sort of template facility in your web
# framework, but here we just use Python string formatting.
html_content = """
<html>
  <head>
    <!-- TODO: There should be a way to include all of the required javascript
               and CSS so matplotlib can add to the set in the future if it
               needs to. -->
    <link rel="stylesheet" href="_static/css/page.css" type="text/css">
    <link rel="stylesheet" href="_static/css/boilerplate.css" type="text/css" />
    <link rel="stylesheet" href="_static/css/fbm.css" type="text/css" />
    <link rel="stylesheet" href="_static/jquery/css/themes/base/jquery-ui.min.css" >
    <script src="_static/jquery/js/jquery-1.7.1.min.js"></script>
    <script src="_static/jquery/js/jquery-ui.min.js"></script>
    <script src="mpl.js"></script>

    <script>
      /* This is a callback that is called when the user saves
         (downloads) a file.  Its purpose is really to map from a
         figure and file format to a url in the application. */
      function ondownload(figure, format) {
        window.open('download.' + format, '_blank');
      };

      $(document).ready(
        function() {
          /* It is up to the application to provide a websocket that the figure
             will use to communicate to the server.  This websocket object can
             also be a "fake" websocket that underneath multiplexes messages
             from multiple figures, if necessary. */
          var websocket_type = mpl.get_websocket_type();
          var websocket = new websocket_type("%(ws_uri)sws");

          // mpl.figure creates a new figure on the webpage.
          var fig = new mpl.figure(
              // A unique numeric identifier for the figure
              %(fig_id)s,
              // A websocket object (or something that behaves like one)
              websocket,
              // A function called when a file type is selected for download
              ondownload,
              // The HTML element in which to place the figure
              $('div#figure'));
        }
      );
    </script>

    <title>matplotlib</title>
  </head>

  <body>
    <div id="figure">
    </div>
  </body>
</html>
"""


class MyApplication(tornado.web.Application):
    class MainPage(tornado.web.RequestHandler):
        """
        Serves the main HTML page.
        """
        def get(self):
            manager = self.application.manager
            ws_uri = "ws://{req.host}/".format(req=self.request)
            content = html_content % {
                "ws_uri": ws_uri, "fig_id": manager.num}
            self.write(content)

    class MplJs(tornado.web.RequestHandler):
        """
        Serves the generated matplotlib javascript file.  The content
        is dynamically generated based on which toolbar functions the
        user has defined.  Call `FigureManagerWebAgg` to get its
        content.
        """
        def get(self):
            self.set_header('Content-Type', 'application/javascript')
            js_content = FigureManagerWebAgg.get_javascript()

            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        """
        Handles downloading of the figure in various file formats.
        """
        def get(self, fmt):
            manager = self.application.manager

            mimetypes = {
                'ps': 'application/postscript',
                'eps': 'application/postscript',
                'pdf': 'application/pdf',
                'svg': 'image/svg+xml',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'tif': 'image/tiff',
                'emf': 'application/emf'
            }

            self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))

            buff = io.BytesIO()
            manager.canvas.print_figure(buff, format=fmt)
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        """
        A websocket for interactive communication between the plot in
        the browser and the server.

        In addition to the methods required by tornado, it is required to
        have two callback methods:

            - ``send_json(json_content)`` is called by matplotlib when
              it needs to send json to the browser.  `json_content` is
              a JSON tree (Python dictionary), and it is the responsibility
              of this implementation to encode it as a string to send over
              the socket.

            - ``send_binary(blob)`` is called to send binary image data
              to the browser.
        """
        supports_binary = True

        def open(self):
            # Register the websocket with the FigureManager.
            manager = self.application.manager
            manager.add_web_socket(self)
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            # When the socket is closed, deregister the websocket with
            # the FigureManager.
            manager = self.application.manager
            manager.remove_web_socket(self)

        def on_message(self, message):
            # The 'supports_binary' message is relevant to the
            # websocket itself.  The other messages get passed along
            # to matplotlib as-is.

            # Every message has a "type" and a "figure_id".
            message = json.loads(message)
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = self.application.manager
                manager.handle_json(message)

        def send_json(self, content):
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = "data:image/png;base64,{0}".format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)

    def __init__(self, figure):
        self.figure = figure
        self.manager = new_figure_manager_given_figure(
            id(figure), figure)

        super(MyApplication, self).__init__([
            # Static files for the CSS and JS
            (r'/_static/(.*)',
             tornado.web.StaticFileHandler,
             {'path': FigureManagerWebAgg.get_static_file_path()}),

            # The page that contains all of the pieces
            ('/', self.MainPage),

            ('/mpl.js', self.MplJs),

            # Sends images and events to the browser, and receives
            # events from the browser
            ('/ws', self.WebSocket),

            # Handles the downloading (i.e., saving) of static images
            (r'/download.([a-z0-9.]+)', self.Download),
        ])


if __name__ == "__main__":
    figure = create_figure()
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()




********************************************
./user_interfaces/embedding_in_qt4.py
#!/usr/bin/env python

# embedding_in_qt4.py --- Simple Qt4 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals
import sys
import os
import random
from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtGui.QMessageBox.about(self, "About",
"""embedding_in_qt4.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale

This program is a simple example of a Qt4 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation."""
)


qApp = QtGui.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()




********************************************
./user_interfaces/svg_histogram.py
#!/usr/bin/env python
#-*- encoding:utf-8 -*-

"""
Demonstrate how to create an interactive histogram, in which bars
are hidden or shown by cliking on legend markers. 

The interactivity is encoded in ecmascript (javascript) and inserted in 
the SVG code in a post-processing step. To render the image, open it in 
a web browser. SVG is supported in most web browsers used by Linux and 
OSX users. Windows IE9 supports SVG, but earlier versions do not. 

Notes
-----
The matplotlib backend lets us assign ids to each object. This is the
mechanism used here to relate matplotlib objects created in python and 
the corresponding SVG constructs that are parsed in the second step.
While flexible, ids are cumbersome to use for large collection of 
objects. Two mechanisms could be used to simplify things:
 * systematic grouping of objects into SVG <g> tags,
 * assingning classes to each SVG object according to its origin.
 
For example, instead of modifying the properties of each individual bar, 
the bars from the `hist` function could either be grouped in
a PatchCollection, or be assigned a class="hist_##" attribute. 

CSS could also be used more extensively to replace repetitive markup
troughout the generated SVG. 

__author__="david.huard@gmail.com"

"""


import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from StringIO import StringIO
import json

plt.rcParams['svg.embed_char_paths'] = 'none'

# Apparently, this `register_namespace` method works only with 
# python 2.7 and up and is necessary to avoid garbling the XML name
# space with ns0.
ET.register_namespace("","http://www.w3.org/2000/svg")




# --- Create histogram, legend and title ---
plt.figure()
r = np.random.randn(100)
r1 = r + 1
labels = ['Rabbits', 'Frogs']
H = plt.hist([r,r1], label=labels)
containers = H[-1]
leg = plt.legend(frameon=False)
plt.title("""From a web browser, click on the legend 
marker to toggle the corresponding histogram.""")


# --- Add ids to the svg objects we'll modify 

hist_patches = {}
for ic, c in enumerate(containers):
    hist_patches['hist_%d'%ic] = []
    for il, element in enumerate(c):
        element.set_gid('hist_%d_patch_%d'%(ic, il))
        hist_patches['hist_%d'%ic].append('hist_%d_patch_%d'%(ic,il))    

# Set ids for the legend patches    
for i, t in enumerate(leg.get_patches()):
    t.set_gid('leg_patch_%d'%i)

# Set ids for the text patches
for i, t in enumerate(leg.get_texts()):
    t.set_gid('leg_text_%d'%i)
            
# Save SVG in a fake file object.
f = StringIO()
plt.savefig(f, format="svg")

# Create XML tree from the SVG file.
tree, xmlid = ET.XMLID(f.getvalue())


# --- Add interactivity ---

# Add attributes to the patch objects.    
for i, t in enumerate(leg.get_patches()):
    el = xmlid['leg_patch_%d'%i]
    el.set('cursor', 'pointer')
    el.set('onclick', "toggle_hist(this)")
    
# Add attributes to the text objects.    
for i, t in enumerate(leg.get_texts()):
    el = xmlid['leg_text_%d'%i]
    el.set('cursor', 'pointer')
    el.set('onclick', "toggle_hist(this)")
        
# Create script defining the function `toggle_hist`. 
# We create a global variable `container` that stores the patches id 
# belonging to each histogram. Then a function "toggle_element" sets the 
# visibility attribute of all patches of each histogram and the opacity
# of the marker itself. 

script = """
<script type="text/ecmascript">
<![CDATA[
var container = %s 

function toggle(oid, attribute, values) {
    /* Toggle the style attribute of an object between two values. 
    
    Parameters
    ----------
    oid : str
      Object identifier.
    attribute : str
      Name of syle attribute.
    values : [on state, off state]
      The two values that are switched between.
    */
    var obj = document.getElementById(oid);
    var a = obj.style[attribute];
    
    a = (a == values[0] || a == "") ? values[1] : values[0];
    obj.style[attribute] = a;
    }

function toggle_hist(obj) {
    
    var num = obj.id.slice(-1);
    
    toggle('leg_patch_' + num, 'opacity', [1, 0.3]);
    toggle('leg_text_' + num, 'opacity', [1, 0.5]);
    
    var names = container['hist_'+num]
    
    for (var i=0; i < names.length; i++) {
        toggle(names[i], 'opacity', [1,0])
    };
    }
]]>
</script>
"""%json.dumps(hist_patches)

# Add a transition effect
css = tree.getchildren()[0][0]
css.text = css.text + "g {-webkit-transition:opacity 0.4s ease-out;-moz-transition:opacity 0.4s ease-out;}"
    
# Insert the script and save to file.
tree.insert(0, ET.XML(script))

ET.ElementTree(tree).write("svg_histogram.svg")

    





********************************************
./user_interfaces/embedding_in_wx3.py
#!/usr/bin/env python

"""
Copyright (C) 2003-2004 Andrew Straw, Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

This is yet another example of using matplotlib with wx.  Hopefully
this is pretty full-featured:

  - both matplotlib toolbar and WX buttons manipulate plot
  - full wxApp framework, including widget interaction
  - XRC (XML wxWidgets resource) file to create GUI (made with XRCed)

This was derived from embedding_in_wx and dynamic_image_wxagg.

Thanks to matplotlib and wx teams for creating such great software!

"""
from __future__ import print_function

# Used to guarantee to use at least Wx2.8
import wxversion
wxversion.ensureMinimal('2.8')

import sys, time, os, gc
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.figure import Figure
import numpy as np

import wx
import wx.xrc as xrc

ERR_TOL = 1e-5 # floating point slop for peak-detection


matplotlib.rc('image', origin='lower')

class PlotPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = Toolbar(self.canvas) #matplotlib toolbar
        self.toolbar.Realize()
        #self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def init_plot_data(self):
        a = self.fig.add_subplot(111)

        x = np.arange(120.0)*2*np.pi/60.0
        y = np.arange(100.0)*2*np.pi/50.0
        self.x, self.y = np.meshgrid(x, y)
        z = np.sin(self.x) + np.cos(self.y)
        self.im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')

        zmax = np.amax(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0]-ymax_i
        self.lines = a.plot(xmax_i,ymax_i,'ko')

        self.toolbar.update() # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def OnWhiz(self,evt):
        self.x += np.pi/15
        self.y += np.pi/20
        z = np.sin(self.x) + np.cos(self.y)
        self.im.set_array(z)

        zmax = np.amax(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0]-ymax_i
        self.lines[0].set_data(xmax_i,ymax_i)

        self.canvas.draw()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

class MyApp(wx.App):
    def OnInit(self):
        xrcfile = cbook.get_sample_data('embedding_in_wx3.xrc', asfileobj=False)
        print('loading', xrcfile)

        self.res = xrc.XmlResource(xrcfile)

        # main frame and panel ---------

        self.frame = self.res.LoadFrame(None,"MainFrame")
        self.panel = xrc.XRCCTRL(self.frame,"MainPanel")

        # matplotlib panel -------------

        # container for matplotlib panel (I like to make a container
        # panel for our panel so I know where it'll go when in XRCed.)
        plot_container = xrc.XRCCTRL(self.frame,"plot_container_panel")
        sizer = wx.BoxSizer(wx.VERTICAL)

        # matplotlib panel itself
        self.plotpanel = PlotPanel(plot_container)
        self.plotpanel.init_plot_data()

        # wx boilerplate
        sizer.Add(self.plotpanel, 1, wx.EXPAND)
        plot_container.SetSizer(sizer)

        # whiz button ------------------

        whiz_button = xrc.XRCCTRL(self.frame,"whiz_button")
        wx.EVT_BUTTON(whiz_button, whiz_button.GetId(),
                      self.plotpanel.OnWhiz)

        # bang button ------------------

        bang_button = xrc.XRCCTRL(self.frame,"bang_button")
        wx.EVT_BUTTON(bang_button, bang_button.GetId(),
                      self.OnBang)

        # final setup ------------------

        sizer = self.panel.GetSizer()
        self.frame.Show(1)

        self.SetTopWindow(self.frame)

        return True

    def OnBang(self,event):
        bang_count = xrc.XRCCTRL(self.frame,"bang_count")
        bangs = bang_count.GetValue()
        bangs = int(bangs)+1
        bang_count.SetValue(str(bangs))

if __name__ == '__main__':
    app = MyApp(0)
    app.MainLoop()





********************************************
./user_interfaces/embedding_in_tk.py
#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Embedding in TK")


f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

a.plot(t,s)


# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg( canvas, root )
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def on_key_event(event):
    print('you pressed %s'%event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.






********************************************
./user_interfaces/svg_tooltip.py
"""
SVG tooltip example
===================

This example shows how to create a tooltip that will show up when 
hovering over a matplotlib patch. 

Although it is possible to create the tooltip from CSS or javascript, 
here we create it in matplotlib and simply toggle its visibility on 
when hovering over the patch. This approach provides total control over 
the tooltip placement and appearance, at the expense of more code up
front. 

The alternative approach would be to put the tooltip content in `title` 
atttributes of SVG objects. Then, using an existing js/CSS library, it 
would be relatively straightforward to create the tooltip in the 
browser. The content would be dictated by the `title` attribute, and 
the appearance by the CSS. 


:author: David Huard
"""


import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from StringIO import StringIO

ET.register_namespace("","http://www.w3.org/2000/svg")

fig, ax = plt.subplots()

# Create patches to which tooltips will be assigned.
circle = plt.Circle((0,0), 5, fc='blue')
rect = plt.Rectangle((-5, 10), 10, 5, fc='green')

ax.add_patch(circle)
ax.add_patch(rect)

# Create the tooltips
circle_tip = ax.annotate('This is a blue circle.', 
            xy=(0,0), 
            xytext=(30,-30), 
            textcoords='offset points', 
            color='w', 
            ha='left', 
            bbox=dict(boxstyle='round,pad=.5', fc=(.1,.1,.1,.92), ec=(1.,1.,1.), lw=1, zorder=1),
            )
            
rect_tip = ax.annotate('This is a green rectangle.', 
            xy=(-5,10), 
            xytext=(30,40), 
            textcoords='offset points', 
            color='w', 
            ha='left', 
            bbox=dict(boxstyle='round,pad=.5', fc=(.1,.1,.1,.92), ec=(1.,1.,1.), lw=1, zorder=1),
            )
            

# Set id for the patches    
for i, t in enumerate(ax.patches):
    t.set_gid('patch_%d'%i)

# Set id for the annotations
for i, t in enumerate(ax.texts):
    t.set_gid('tooltip_%d'%i)


# Save the figure in a fake file object
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')

f = StringIO()
plt.savefig(f, format="svg")

# --- Add interactivity ---

# Create XML tree from the SVG file.
tree, xmlid = ET.XMLID(f.getvalue())
tree.set('onload', 'init(evt)')

# Hide the tooltips
for i, t in enumerate(ax.texts):
    el = xmlid['tooltip_%d'%i]
    el.set('visibility', 'hidden')

# Assign onmouseover and onmouseout callbacks to patches.        
for i, t in enumerate(ax.patches):
    el = xmlid['patch_%d'%i]
    el.set('onmouseover', "ShowTooltip(this)")
    el.set('onmouseout', "HideTooltip(this)")

# This is the script defining the ShowTooltip and HideTooltip functions.
script = """
    <script type="text/ecmascript">
    <![CDATA[
    
    function init(evt) {
        if ( window.svgDocument == null ) {
            svgDocument = evt.target.ownerDocument;
            }
        }
        
    function ShowTooltip(obj) {
        var cur = obj.id.slice(-1);
        
        var tip = svgDocument.getElementById('tooltip_' + cur);
        tip.setAttribute('visibility',"visible")
        }
        
    function HideTooltip(obj) {
        var cur = obj.id.slice(-1);
        var tip = svgDocument.getElementById('tooltip_' + cur);
        tip.setAttribute('visibility',"hidden")
        }
        
    ]]>
    </script>
    """

# Insert the script at the top of the file and save it.
tree.insert(0, ET.XML(script))
ET.ElementTree(tree).write('svg_tooltip.svg')




********************************************
./user_interfaces/embedding_in_gtk2.py
#!/usr/bin/env python
"""
show how to add a matplotlib FigureCanvasGTK or FigureCanvasGTKAgg widget and
a toolbar to a gtk.Window
"""
import gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi

# uncomment to select /GTK/GTKAgg/GTKCairo
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas

# or NavigationToolbar for classic
#from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

win = gtk.Window()
win.connect("destroy", lambda x: gtk.main_quit())
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

vbox = gtk.VBox()
win.add(vbox)

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

ax.plot(t,s)


canvas = FigureCanvas(fig)  # a gtk.DrawingArea
vbox.pack_start(canvas)
toolbar = NavigationToolbar(canvas, win)
vbox.pack_start(toolbar, False, False)


def on_key_event(event):
    print('you pressed %s'%event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

win.show_all()
gtk.main()




********************************************
./user_interfaces/wxcursor_demo.py
"""
Example to draw a cursor and report the data coords in wx
"""

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from numpy import arange, sin, pi

import wx

class CanvasFrame(wx.Frame):

    def __init__(self, ):
        wx.Frame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)

        self.axes.plot(t,s)
        self.axes.set_xlabel('t')
        self.axes.set_ylabel('sin(t)')
        self.figure_canvas = FigureCanvas(self, -1, self.figure)

        # Note that event is a MplEvent
        self.figure_canvas.mpl_connect('motion_notify_event', self.UpdateStatusBar)
        self.figure_canvas.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.figure_canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.statusBar = wx.StatusBar(self, -1)
        self.statusBar.SetFieldsCount(1)
        self.SetStatusBar(self.statusBar)

        self.toolbar = NavigationToolbar2Wx(self.figure_canvas)
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.Show()

    def ChangeCursor(self, event):
        self.figure_canvas.SetCursor(wx.StockCursor(wx.CURSOR_BULLSEYE))

    def UpdateStatusBar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusBar.SetStatusText(( "x= " + str(x) +
                                           "  y=" +str(y) ),
                                           0)

class App(wx.App):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        self.SetTopWindow(frame)
        frame.Show(True)
        return True

if __name__=='__main__':
    app = App(0)
    app.MainLoop()




********************************************
./user_interfaces/mpl_with_glade.py
#!/usr/bin/env python

from __future__ import print_function
import matplotlib
matplotlib.use('GTK')

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.widgets import SpanSelector

from numpy import arange, sin, pi
import gtk
import gtk.glade


def simple_msg(msg, parent=None, title=None):
    dialog = gtk.MessageDialog(
        parent         = None,
        type           = gtk.MESSAGE_INFO,
        buttons        = gtk.BUTTONS_OK,
        message_format = msg)
    if parent is not None:
        dialog.set_transient_for(parent)
    if title is not None:
        dialog.set_title(title)
    dialog.show()
    dialog.run()
    dialog.destroy()
    return None



class GladeHandlers:
    def on_buttonClickMe_clicked(event):
        simple_msg('Nothing to say, really',
                   parent=widgets['windowMain'],
                   title='Thanks!')

class WidgetsWrapper:
    def __init__(self):
        self.widgets = gtk.glade.XML('mpl_with_glade.glade')
        self.widgets.signal_autoconnect(GladeHandlers.__dict__)

        self['windowMain'].connect('destroy', lambda x: gtk.main_quit())
        self['windowMain'].move(10,10)
        self.figure = Figure(figsize=(8,6), dpi=72)
        self.axis = self.figure.add_subplot(111)

        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        self.axis.plot(t,s)
        self.axis.set_xlabel('time (s)')
        self.axis.set_ylabel('voltage')

        self.canvas = FigureCanvas(self.figure) # a gtk.DrawingArea
        self.canvas.show()
        self.canvas.set_size_request(600, 400)
        self.canvas.set_events(
            gtk.gdk.BUTTON_PRESS_MASK      |
            gtk.gdk.KEY_PRESS_MASK      |
            gtk.gdk.KEY_RELEASE_MASK
            )
        self.canvas.set_flags(gtk.HAS_FOCUS|gtk.CAN_FOCUS)
        self.canvas.grab_focus()

        def keypress(widget, event):
            print('key press')
        def buttonpress(widget, event):
            print('button press')

        self.canvas.connect('key_press_event', keypress)
        self.canvas.connect('button_press_event', buttonpress)

        def onselect(xmin, xmax):
            print(xmin, xmax)

        span = SpanSelector(self.axis, onselect, 'horizontal', useblit=False,
                            rectprops=dict(alpha=0.5, facecolor='red') )


        self['vboxMain'].pack_start(self.canvas, True, True)
        self['vboxMain'].show()

        # below is optional if you want the navigation toolbar
        self.navToolbar = NavigationToolbar(self.canvas, self['windowMain'])
        self.navToolbar.lastDir = '/var/tmp/'
        self['vboxMain'].pack_start(self.navToolbar)
        self.navToolbar.show()

        sep = gtk.HSeparator()
        sep.show()
        self['vboxMain'].pack_start(sep, True, True)


        self['vboxMain'].reorder_child(self['buttonClickMe'],-1)





    def __getitem__(self, key):
        return self.widgets.get_widget(key)

widgets = WidgetsWrapper()
gtk.main()




********************************************
./user_interfaces/embedding_in_wx5.py
# Used to guarantee to use at least Wx2.8
import wxversion
wxversion.ensureMinimal('2.8')

import wx
import wx.aui
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

class Plot(wx.Panel):
    def __init__(self, parent, id = -1, dpi = None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2,2))
        self.canvas = Canvas(self, -1, self.figure)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.EXPAND)
        sizer.Add(self.toolbar, 0 , wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

class PlotNotebook(wx.Panel):
    def __init__(self, parent, id = -1):
        wx.Panel.__init__(self, parent, id=id)
        self.nb = wx.aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self,name="plot"):
       page = Plot(self.nb)
       self.nb.AddPage(page,name)
       return page.figure


def demo():
    app = wx.PySimpleApp()
    frame = wx.Frame(None,-1,'Plotter')
    plotter = PlotNotebook(frame)
    axes1 = plotter.add('figure 1').gca()
    axes1.plot([1,2,3],[2,1,4])
    axes2 = plotter.add('figure 2').gca()
    axes2.plot([1,2,3,4,5],[2,1,4,2,3])
    #axes1.figure.canvas.draw()
    #axes2.figure.canvas.draw()
    frame.Show()
    app.MainLoop()

if __name__ == "__main__": demo()





********************************************
./user_interfaces/lineprops_dialog_gtk.py
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib.backends.backend_gtk import DialogLineprops

import numpy as np
import matplotlib.pyplot as plt

def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1,e1)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

fig, ax = plt.subplots()
l1,  = ax.plot(t1, f(t1), 'bo', label='line 1')
l2,  = ax.plot(t2, f(t2), 'k--', label='line 2')

dlg = DialogLineprops([l1,l2])
dlg.show()
plt.show()





********************************************
./user_interfaces/interactive.py
#!/usr/bin/env python

"""Multithreaded interactive interpreter with GTK and Matplotlib support.

WARNING:
As of 2010/06/25, this is not working, at least on Linux.
I have disabled it as a runnable script. - EF


Usage:

  pyint-gtk.py -> starts shell with gtk thread running separately

  pyint-gtk.py -pylab [filename] -> initializes matplotlib, optionally running
  the named file.  The shell starts after the file is executed.

Threading code taken from:
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/65109, by Brian
McErlean and John Finlay.

Matplotlib support taken from interactive.py in the matplotlib distribution.

Also borrows liberally from code.py in the Python standard library."""

from __future__ import print_function

__author__ = "Fernando Perez <Fernando.Perez@colorado.edu>"

import sys
import code
import threading

import gobject
import gtk

try:
    import readline
except ImportError:
    has_readline = False
else:
    has_readline = True

class MTConsole(code.InteractiveConsole):
    """Simple multi-threaded shell"""

    def __init__(self,on_kill=None,*args,**kw):
        code.InteractiveConsole.__init__(self,*args,**kw)
        self.code_to_run = None
        self.ready = threading.Condition()
        self._kill = False
        if on_kill is None:
            on_kill = []
        # Check that all things to kill are callable:
        for _ in on_kill:
            if not callable(_):
                raise TypeError,'on_kill must be a list of callables'
        self.on_kill = on_kill
        # Set up tab-completer
        if has_readline:
            import rlcompleter
            try:  # this form only works with python 2.3
                self.completer = rlcompleter.Completer(self.locals)
            except: # simpler for py2.2
                self.completer = rlcompleter.Completer()

            readline.set_completer(self.completer.complete)
            # Use tab for completions
            readline.parse_and_bind('tab: complete')
            # This forces readline to automatically print the above list when tab
            # completion is set to 'complete'.
            readline.parse_and_bind('set show-all-if-ambiguous on')
            # Bindings for incremental searches in the history. These searches
            # use the string typed so far on the command line and search
            # anything in the previous input history containing them.
            readline.parse_and_bind('"\C-r": reverse-search-history')
            readline.parse_and_bind('"\C-s": forward-search-history')

    def runsource(self, source, filename="<input>", symbol="single"):
        """Compile and run some source in the interpreter.

        Arguments are as for compile_command().

        One several things can happen:

        1) The input is incorrect; compile_command() raised an
        exception (SyntaxError or OverflowError).  A syntax traceback
        will be printed by calling the showsyntaxerror() method.

        2) The input is incomplete, and more input is required;
        compile_command() returned None.  Nothing happens.

        3) The input is complete; compile_command() returned a code
        object.  The code is executed by calling self.runcode() (which
        also handles run-time exceptions, except for SystemExit).

        The return value is True in case 2, False in the other cases (unless
        an exception is raised).  The return value can be used to
        decide whether to use sys.ps1 or sys.ps2 to prompt the next
        line.
        """
        try:
            code = self.compile(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1
            self.showsyntaxerror(filename)
            return False

        if code is None:
            # Case 2
            return True

        # Case 3
        # Store code in self, so the execution thread can handle it
        self.ready.acquire()
        self.code_to_run = code
        self.ready.wait()  # Wait until processed in timeout interval
        self.ready.release()

        return False

    def runcode(self):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to display a
        traceback."""

        self.ready.acquire()
        if self._kill:
            print('Closing threads...')
            sys.stdout.flush()
            for tokill in self.on_kill:
                tokill()
            print('Done.')

        if self.code_to_run is not None:
            self.ready.notify()
            code.InteractiveConsole.runcode(self,self.code_to_run)

        self.code_to_run = None
        self.ready.release()
        return True

    def kill (self):
        """Kill the thread, returning when it has been shut down."""
        self.ready.acquire()
        self._kill = True
        self.ready.release()

class GTKInterpreter(threading.Thread):
    """Run gtk.main in the main thread and a python interpreter in a
    separate thread.
    Python commands can be passed to the thread where they will be executed.
    This is implemented by periodically checking for passed code using a
    GTK timeout callback.
    """
    TIMEOUT = 100 # Millisecond interval between timeouts.

    def __init__(self,banner=None):
        threading.Thread.__init__(self)
        self.banner = banner
        self.shell = MTConsole(on_kill=[gtk.main_quit])

    def run(self):
        self.pre_interact()
        self.shell.interact(self.banner)
        self.shell.kill()

    def mainloop(self):
        self.start()
        gobject.timeout_add(self.TIMEOUT, self.shell.runcode)
        try:
            if gtk.gtk_version[0] >= 2:
                gtk.gdk.threads_init()
        except AttributeError:
            pass
        gtk.main()
        self.join()

    def pre_interact(self):
        """This method should be overridden by subclasses.

        It gets called right before interact(), but after the thread starts.
        Typically used to push initialization code into the interpreter"""

        pass

class MatplotLibInterpreter(GTKInterpreter):
    """Threaded interpreter with matplotlib support.

    Note that this explicitly sets GTKAgg as the backend, since it has
    specific GTK hooks in it."""

    def __init__(self,banner=None):
        banner = """\nWelcome to matplotlib, a MATLAB-like python environment.
    help(matlab)   -> help on matlab compatible commands from matplotlib.
    help(plotting) -> help on plotting commands.
    """
        GTKInterpreter.__init__(self,banner)

    def pre_interact(self):
        """Initialize matplotlib before user interaction begins"""

        push = self.shell.push
        # Code to execute in user's namespace
        lines = ["import matplotlib",
                 "matplotlib.use('GTKAgg')",
                 "matplotlib.interactive(1)",
                 "import matplotlib.pylab as pylab",
                 "from matplotlib.pylab import *\n"]

        map(push,lines)

        # Execute file if given.
        if len(sys.argv)>1:
            import matplotlib
            matplotlib.interactive(0) # turn off interaction
            fname = sys.argv[1]
            try:
                inFile = file(fname, 'r')
            except IOError:
                print('*** ERROR *** Could not read file <%s>' % fname)
            else:
                print('*** Executing file <%s>:' % fname)
                for line in inFile:
                    if line.lstrip().find('show()')==0: continue
                    print('>>', line)
                    push(line)
                inFile.close()
            matplotlib.interactive(1)   # turn on interaction

if __name__ == '__main__':
    print("This demo is not presently functional, so running")
    print("it as a script has been disabled.")
    sys.exit()
    # Quick sys.argv hack to extract the option and leave filenames in sys.argv.
    # For real option handling, use optparse or getopt.
    if len(sys.argv) > 1 and sys.argv[1]=='-pylab':
        sys.argv = [sys.argv[0]]+sys.argv[2:]
        MatplotLibInterpreter().mainloop()
    else:
        GTKInterpreter().mainloop()




********************************************
./api/line_with_text.py
"""
Show how to override basic methods so an artist can contain another
artist.  In this case, the line contains a Text instance to label it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext


class MyLine(lines.Line2D):

   def __init__(self, *args, **kwargs):
      # we'll update the position when the line data is set
      self.text = mtext.Text(0, 0, '')
      lines.Line2D.__init__(self, *args, **kwargs)

      # we can't access the label attr until *after* the line is
      # inited
      self.text.set_text(self.get_label())

   def set_figure(self, figure):
      self.text.set_figure(figure)
      lines.Line2D.set_figure(self, figure)

   def set_axes(self, axes):
      self.text.set_axes(axes)
      lines.Line2D.set_axes(self, axes)

   def set_transform(self, transform):
      # 2 pixel offset
      texttrans = transform + mtransforms.Affine2D().translate(2, 2)
      self.text.set_transform(texttrans)
      lines.Line2D.set_transform(self, transform)


   def set_data(self, x, y):
      if len(x):
         self.text.set_position((x[-1], y[-1]))

      lines.Line2D.set_data(self, x, y)

   def draw(self, renderer):
      # draw my label at the end of the line with 2 pixel offset
      lines.Line2D.draw(self, renderer)
      self.text.draw(renderer)





fig, ax = plt.subplots()
x, y = np.random.rand(2, 20)
line = MyLine(x, y, mfc='red', ms=12, label='line label')
#line.text.set_text('line label')
line.text.set_color('red')
line.text.set_fontsize(16)


ax.add_line(line)


plt.show()




********************************************
./api/watermark_text.py
"""
Use a Text as a watermark
"""
import numpy as np
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()

# position bottom right
fig.text(0.95, 0.05, 'Property of MPL',
         fontsize=50, color='gray',
         ha='right', va='bottom', alpha=0.5)

plt.show()




********************************************
./api/two_scales.py
#!/usr/bin/env python
"""

Demonstrate how to do two plots on the same axes with different left
right scales.


The trick is to use *2 different axes*.  Turn the axes rectangular
frame off on the 2nd axes to keep it from obscuring the first.
Manually set the tick locs and labels as desired.  You can use
separate matplotlib.ticker formatters and locators as desired since
the two axes are independent.

This is achieved in the following example by calling the Axes.twinx()
method, which performs this work. See the source of twinx() in
axes.py for an example of how to do it for different x scales. (Hint:
use the xaxis instance and call tick_bottom and tick_top in place of
tick_left and tick_right.)

The twinx and twiny methods are also exposed as pyplot functions.

"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('exp', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')


ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('sin', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()





********************************************
./api/colorbar_only.py
'''
Make a colorbar as a separate figure.
'''

from matplotlib import pyplot
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = pyplot.figure(figsize=(8,3))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
cb1.set_label('Some Units')

# The second example illustrates the use of a ListedColormap, a
# BoundaryNorm, and extended ends to show the "over" and "under"
# value colors.
cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 4, 7, 8]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     boundaries=[0]+bounds+[13],
                                     extend='both',
                                     ticks=bounds, # optional
                                     spacing='proportional',
                                     orientation='horizontal')
cb2.set_label('Discrete intervals, some other units')

# The third example illustrates the use of custom length colorbar
# extensions, used on a colorbar with discrete intervals.
cmap = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
    [1., .8, 0.], [1., .4, 0.]])
cmap.set_over((1., 0., 0.))
cmap.set_under((0., 0., 1.))

bounds = [-1., -.5, 0., .5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                     norm=norm,
                                     boundaries=[-10]+bounds+[10],
                                     extend='both',
                                     # Make the length of each extension
                                     # the same as the length of the
                                     # interior colors:
                                     extendfrac='auto',
                                     ticks=bounds,
                                     spacing='uniform',
                                     orientation='horizontal')
cb3.set_label('Custom extension lengths, some other units')

pyplot.show()




********************************************
./api/scatter_piecharts.py
"""
This example makes custom 'pie charts' as the markers for a scatter plotqu

Thanks to Manuel Metz for the example
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# first define the ratios
r1 = 0.2      # 20%
r2 = r1 + 0.4 # 40%

# define some sizes of the scatter marker
sizes = [60,80,120]

# calculate the points of the first pie marker
#
# these are just the origin (0,0) +
# some points on a circle cos,sin
x = [0] + np.cos(np.linspace(0, 2*math.pi*r1, 10)).tolist()
y = [0] + np.sin(np.linspace(0, 2*math.pi*r1, 10)).tolist()
xy1 = list(zip(x,y))

# ...
x = [0] + np.cos(np.linspace(2*math.pi*r1, 2*math.pi*r2, 10)).tolist()
y = [0] + np.sin(np.linspace(2*math.pi*r1, 2*math.pi*r2, 10)).tolist()
xy2 = list(zip(x,y))

x = [0] + np.cos(np.linspace(2*math.pi*r2, 2*math.pi, 10)).tolist()
y = [0] + np.sin(np.linspace(2*math.pi*r2, 2*math.pi, 10)).tolist()
xy3 = list(zip(x,y))


fig, ax = plt.subplots()
ax.scatter( np.arange(3), np.arange(3), marker=(xy1,0), s=sizes, facecolor='blue' )
ax.scatter( np.arange(3), np.arange(3), marker=(xy2,0), s=sizes, facecolor='green' )
ax.scatter( np.arange(3), np.arange(3), marker=(xy3,0), s=sizes, facecolor='red' )
plt.show()




********************************************
./api/sankey_demo_basics.py
"""Demonstrate the Sankey class by producing three basic diagrams.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.sankey import Sankey


# Example 1 -- Mostly defaults
# This demonstrates how to create a simple diagram by implicitly calling the
# Sankey.add() method and by appending finish() to the call to the class.
Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
       labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
       orientations=[-1, 1, 0, 1, 1, 1, 0, -1]).finish()
plt.title("The default settings produce a diagram like this.")
# Notice:
#   1. Axes weren't provided when Sankey() was instantiated, so they were
#      created automatically.
#   2. The scale argument wasn't necessary since the data was already
#      normalized.
#   3. By default, the lengths of the paths are justified.

# Example 2
# This demonstrates:
#   1. Setting one path longer than the others
#   2. Placing a label in the middle of the diagram
#   3. Using the the scale argument to normalize the flows
#   4. Implicitly passing keyword arguments to PathPatch()
#   5. Changing the angle of the arrow heads
#   6. Changing the offset between the tips of the paths and their labels
#   7. Formatting the numbers in the path labels and the associated unit
#   8. Changing the appearance of the patch and the labels after the figure is
#      created
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Flow Diagram of a Widget")
sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
                format='%.0f', unit='%')
sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
           labels = ['', '', '', 'First', 'Second', 'Third', 'Fourth',
                     'Fifth', 'Hurray!'],
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
           pathlengths = [0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
                          0.25],
           patchlabel="Widget\nA",
           alpha=0.2, lw=2.0) # Arguments to matplotlib.patches.PathPatch()
diagrams = sankey.finish()
diagrams[0].patch.set_facecolor('#37c959')
diagrams[0].texts[-1].set_color('r')
diagrams[0].text.set_fontweight('bold')
# Notice:
#   1. Since the sum of the flows is nonzero, the width of the trunk isn't
#      uniform.  If verbose.level is helpful (in matplotlibrc), a message is
#      given in the terminal window.
#   2. The second flow doesn't appear because its value is zero.  Again, if
#      verbose.level is helpful, a message is given in the terminal window.

# Example 3
# This demonstrates:
#   1. Connecting two systems
#   2. Turning off the labels of the quantities
#   3. Adding a legend
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
sankey = Sankey(ax=ax, unit=None)
sankey.add(flows=flows, label='one',
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
sankey.add(flows=[-0.25, 0.15, 0.1], fc='#37c959', label='two',
           orientations=[-1, -1, -1], prior=0, connect=(0, 0))
diagrams = sankey.finish()
diagrams[-1].patch.set_hatch('/')
plt.legend(loc='best')
# Notice that only one connection is specified, but the systems form a
# circuit since: (1) the lengths of the paths are justified and (2) the
# orientation and ordering of the flows is mirrored.

plt.show()




********************************************
./api/fahrenheit_celsius_scales.py
"""
Show how to display two scales on the left and right y axis -- Fahrenheit and Celsius
"""

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()  # ax1 is the Fahrenheit scale
ax2 = ax1.twinx()          # ax2 is the Celsius scale

def Tc(Tf):
    return (5./9.)*(Tf-32)


def update_ax2(ax1):
   y1, y2 = ax1.get_ylim()
   ax2.set_ylim(Tc(y1), Tc(y2))
   ax2.figure.canvas.draw()

# automatically update ylim of ax2 when ylim of ax1 changes.
ax1.callbacks.connect("ylim_changed", update_ax2)
ax1.plot([78, 79, 79, 77])

ax1.set_title('Two scales: Fahrenheit and Celsius')
ax1.set_ylabel('Fahrenheit')
ax2.set_ylabel('Celsius')

plt.show()




********************************************
./api/date_index_formatter.py
"""
When plotting time series, eg financial time series, one often wants
to leave out days on which there is no data, eh weekends.  The example
below shows how to use an 'index formatter' to achieve the desired plot
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker

datafile = cbook.get_sample_data('aapl.csv', asfileobj=False)
print ('loading %s' % datafile)
r = mlab.csv2rec(datafile)

r.sort()
r = r[-30:]  # get the last 30 days


# first we'll do it the default way, with gaps on weekends
fig, ax = plt.subplots()
ax.plot(r.date, r.adj_close, 'o-')
fig.autofmt_xdate()

# next we'll write a custom formatter
N = len(r)
ind = np.arange(N)  # the evenly spaced plot indices

def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N-1)
    return r.date[thisind].strftime('%Y-%m-%d')

fig, ax = plt.subplots()
ax.plot(ind, r.adj_close, 'o-')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()

plt.show()




********************************************
./api/histogram_path_demo.py
"""
This example shows how to use a path patch to draw a bunch of
rectangles.  The technique of using lots of Rectangle instances, or
the faster method of using PolyCollections, were implemented before we
had proper paths with moveto/lineto, closepoly etc in mpl.  Now that
we have them, we can draw collections of regularly shaped objects with
homogeous properties more efficiently with a PathCollection.  This
example makes a histogram -- its more work to set up the vertex arrays
at the outset, but it should be much faster for large numbers of
objects
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

fig, ax = plt.subplots()

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 50)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n


# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
ax.add_patch(patch)

# update the view limits
ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

plt.show()




********************************************
./api/donut_demo.py
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def wise(v):
    if v == 1:
        return "CCW"
    else:
        return "CW"

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))

Path = mpath.Path

fig, ax = plt.subplots()

inside_vertices = make_circle(0.5)
outside_vertices = make_circle(1.0)
codes = np.ones(len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
codes[0] = mpath.Path.MOVETO

for i, (inside, outside) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    # Concatenate the inside and outside subpaths together, changing their
    # order as needed
    vertices = np.concatenate((outside_vertices[::outside],
                               inside_vertices[::inside]))
    # Shift the path
    vertices[:, 0] += i * 2.5
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    all_codes = np.concatenate((codes, codes))
    # Create the Path object
    path = mpath.Path(vertices, all_codes)
    # Add plot it
    patch = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')
    ax.add_patch(patch)

    ax.annotate("Outside %s,\nInside %s" % (wise(outside), wise(inside)),
                (i * 2.5, -1.5), va="top", ha="center")

ax.set_xlim(-2,10)
ax.set_ylim(-3,2)
ax.set_title('Mmm, donuts!')
ax.set_aspect(1.0)
plt.show()






********************************************
./api/agg_oo.py
#!/usr/bin/env python
# -*- noplot -*-
"""
A pure OO (look Ma, no pylab!) example using the agg backend
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.plot([1,2,3])
ax.set_title('hi mom')
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
canvas.print_figure('test')




********************************************
./api/quad_bezier.py
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()
pp1 = mpatches.PathPatch(
    Path([(0, 0), (1, 0), (1, 1), (0, 0)],
         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)
ax.plot([0.75], [0.25], "ro")
ax.set_title('The red point should be on the path')

plt.show()





********************************************
./api/sankey_demo_old.py
#!/usr/bin/env python

from __future__ import print_function

__author__ = "Yannick Copin <ycopin@ipnl.in2p3.fr>"
__version__ = "Time-stamp: <10/02/2010 16:49 ycopin@lyopc548.in2p3.fr>"

import numpy as np


def sankey(ax,
           outputs=[100.], outlabels=None,
           inputs=[100.], inlabels='',
           dx=40, dy=10, outangle=45, w=3, inangle=30, offset=2, **kwargs):
    """Draw a Sankey diagram.

    outputs: array of outputs, should sum up to 100%
    outlabels: output labels (same length as outputs),
    or None (use default labels) or '' (no labels)
    inputs and inlabels: similar for inputs
    dx: horizontal elongation
    dy: vertical elongation
    outangle: output arrow angle [deg]
    w: output arrow shoulder
    inangle: input dip angle
    offset: text offset
    **kwargs: propagated to Patch (e.g., fill=False)

    Return (patch,[intexts,outtexts]).
    """
    import matplotlib.patches as mpatches
    from matplotlib.path import Path

    outs = np.absolute(outputs)
    outsigns = np.sign(outputs)
    outsigns[-1] = 0  # Last output

    ins = np.absolute(inputs)
    insigns = np.sign(inputs)
    insigns[0] = 0  # First input

    assert sum(outs) == 100, "Outputs don't sum up to 100%"
    assert sum(ins) == 100, "Inputs don't sum up to 100%"

    def add_output(path, loss, sign=1):
        h = (loss/2 + w)*np.tan(outangle/180. * np.pi)  # Arrow tip height
        move, (x, y) = path[-1]  # Use last point as reference
        if sign == 0:  # Final loss (horizontal)
            path.extend([(Path.LINETO, [x+dx, y]),
                         (Path.LINETO, [x+dx, y+w]),
                         (Path.LINETO, [x+dx+h, y-loss/2]),  # Tip
                         (Path.LINETO, [x+dx, y-loss-w]),
                         (Path.LINETO, [x+dx, y-loss])])
            outtips.append((sign, path[-3][1]))
        else:  # Intermediate loss (vertical)
            path.extend([(Path.CURVE4, [x+dx/2, y]),
                         (Path.CURVE4, [x+dx, y]),
                         (Path.CURVE4, [x+dx, y+sign*dy]),
                         (Path.LINETO, [x+dx-w, y+sign*dy]),
                         (Path.LINETO, [x+dx+loss/2, y+sign*(dy+h)]),  # Tip
                         (Path.LINETO, [x+dx+loss+w, y+sign*dy]),
                         (Path.LINETO, [x+dx+loss, y+sign*dy]),
                         (Path.CURVE3, [x+dx+loss, y-sign*loss]),
                         (Path.CURVE3, [x+dx/2+loss, y-sign*loss])])
            outtips.append((sign, path[-5][1]))

    def add_input(path, gain, sign=1):
        h = (gain/2)*np.tan(inangle/180. * np.pi)  # Dip depth
        move, (x, y) = path[-1]  # Use last point as reference
        if sign == 0:  # First gain (horizontal)
            path.extend([(Path.LINETO, [x-dx, y]),
                         (Path.LINETO, [x-dx+h, y+gain/2]),  # Dip
                         (Path.LINETO, [x-dx, y+gain])])
            xd, yd = path[-2][1]  # Dip position
            indips.append((sign, [xd-h, yd]))
        else:  # Intermediate gain (vertical)
            path.extend([(Path.CURVE4, [x-dx/2, y]),
                         (Path.CURVE4, [x-dx, y]),
                         (Path.CURVE4, [x-dx, y+sign*dy]),
                         (Path.LINETO, [x-dx-gain/2, y+sign*(dy-h)]),  # Dip
                         (Path.LINETO, [x-dx-gain, y+sign*dy]),
                         (Path.CURVE3, [x-dx-gain, y-sign*gain]),
                         (Path.CURVE3, [x-dx/2-gain, y-sign*gain])])
            xd, yd = path[-4][1]  # Dip position
            indips.append((sign, [xd, yd+sign*h]))

    outtips = []  # Output arrow tip dir. and positions
    urpath = [(Path.MOVETO, [0, 100])]  # 1st point of upper right path
    lrpath = [(Path.LINETO, [0, 0])]  # 1st point of lower right path
    for loss, sign in zip(outs, outsigns):
        add_output(sign>=0 and urpath or lrpath, loss, sign=sign)

    indips = []  # Input arrow tip dir. and positions
    llpath = [(Path.LINETO, [0, 0])]  # 1st point of lower left path
    ulpath = [(Path.MOVETO, [0, 100])]  # 1st point of upper left path
    for gain, sign in reversed(list(zip(ins, insigns))):
        add_input(sign<=0 and llpath or ulpath, gain, sign=sign)

    def revert(path):
        """A path is not just revertable by path[::-1] because of Bezier
        curves."""
        rpath = []
        nextmove = Path.LINETO
        for move, pos in path[::-1]:
            rpath.append((nextmove, pos))
            nextmove = move
        return rpath

    # Concatenate subpathes in correct order
    path = urpath + revert(lrpath) + llpath + revert(ulpath)

    codes, verts = zip(*path)
    verts = np.array(verts)

    # Path patch
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, **kwargs)
    ax.add_patch(patch)

    if False:  # DEBUG
        print("urpath", urpath)
        print("lrpath", revert(lrpath))
        print("llpath", llpath)
        print("ulpath", revert(ulpath))
        xs, ys = zip(*verts)
        ax.plot(xs, ys, 'go-')

    # Labels

    def set_labels(labels, values):
        """Set or check labels according to values."""
        if labels == '':  # No labels
            return labels
        elif labels is None:  # Default labels
            return ['%2d%%' % val for val in values]
        else:
            assert len(labels) == len(values)
            return labels

    def put_labels(labels, positions, output=True):
        """Put labels to positions."""
        texts = []
        lbls = output and labels or labels[::-1]
        for i, label in enumerate(lbls):
            s, (x, y) = positions[i]  # Label direction and position
            if s == 0:
                t = ax.text(x+offset, y, label,
                            ha=output and 'left' or 'right', va='center')
            elif s > 0:
                t = ax.text(x, y+offset, label, ha='center', va='bottom')
            else:
                t = ax.text(x, y-offset, label, ha='center', va='top')
            texts.append(t)
        return texts

    outlabels = set_labels(outlabels, outs)
    outtexts = put_labels(outlabels, outtips, output=True)

    inlabels = set_labels(inlabels, ins)
    intexts = put_labels(inlabels, indips, output=False)

    # Axes management
    ax.set_xlim(verts[:, 0].min()-dx, verts[:, 0].max()+dx)
    ax.set_ylim(verts[:, 1].min()-dy, verts[:, 1].max()+dy)
    ax.set_aspect('equal', adjustable='datalim')

    return patch, [intexts, outtexts]


if __name__=='__main__':

    import matplotlib.pyplot as plt

    outputs = [10., -20., 5., 15., -10., 40.]
    outlabels = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Hurray!']
    outlabels = [s+'\n%d%%' % abs(l) for l, s in zip(outputs, outlabels)]

    inputs = [60., -25., 15.]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Sankey diagram")

    patch, (intexts, outtexts) = sankey(ax, outputs=outputs,
                                        outlabels=outlabels, inputs=inputs,
                                        inlabels=None, fc='g', alpha=0.2)
    outtexts[1].set_color('r')
    outtexts[-1].set_fontweight('bold')

    plt.show()




********************************************
./api/custom_scale_example.py
from __future__ import unicode_literals

import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator


class MercatorLatitudeScale(mscale.ScaleBase):
    """
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a Mercator projection.

    The scale function:
      ln(tan(y) + sec(y))

    The inverse scale function:
      atan(sinh(y))

    Since the Mercator scale tends to infinity at +/- 90 degrees,
    there is user-defined threshold, above and below which nothing
    will be plotted.  This defaults to +/- 85 degrees.

    source:
    http://en.wikipedia.org/wiki/Mercator_projection
    """

    # The scale class must have a member ``name`` that defines the
    # string used to select the scale.  For example,
    # ``gca().set_yscale("mercator")`` would be used to select this
    # scale.
    name = 'mercator'


    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.

        thresh: The degree above which to crop the data.
        """
        mscale.ScaleBase.__init__(self)
        thresh = kwargs.pop("thresh", (85 / 180.0) * np.pi)
        if thresh >= np.pi / 2.0:
            raise ValueError("thresh must be less than pi/2")
        self.thresh = thresh

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.MercatorLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in ``ticker.py``.

        In our case, the Mercator example uses a fixed locator from
        -90 to 90 degrees and a custom formatter class to put convert
        the radians to degrees and put a degree symbol after the
        value::
        """
        class DegreeFormatter(Formatter):
            def __call__(self, x, pos=None):
                # \u00b0 : degree symbol
                return "%d\u00b0" % ((x / np.pi) * 180.0)

        deg2rad = np.pi / 180.0
        axis.set_major_locator(FixedLocator(
                np.arange(-90, 90, 10) * deg2rad))
        axis.set_major_formatter(DegreeFormatter())
        axis.set_minor_formatter(DegreeFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, -self.thresh), min(vmax, self.thresh)

    class MercatorLatitudeTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the Mercator scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1.0 / ma.cos(masked)))
            else:
                return np.log(np.abs(np.tan(a) + 1.0 / np.cos(a)))

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return np.arctan(np.sinh(a))

        def inverted(self):
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)

# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(MercatorLatitudeScale)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = np.arange(-180.0, 180.0, 0.1)
    s = t / 360.0 * np.pi

    plt.plot(t, s, '-', lw=2)
    plt.gca().set_yscale('mercator')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mercator: Projection of the Oppressor')
    plt.grid(True)

    plt.show()





********************************************
./api/compound_path.py
"""
Make a compund path -- in this case two simple polygons, a rectangle
and a triangle.  Use CLOSEOPOLY and MOVETO for the different parts of
the compound path
"""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt


vertices = []
codes = []

codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
vertices = [(1,1), (1,2), (2, 2), (2, 1), (0,0)]

codes += [Path.MOVETO] + [Path.LINETO]*2 + [Path.CLOSEPOLY]
vertices += [(4,4), (5,5), (5, 4), (0,0)]

vertices = np.array(vertices, float)
path = Path(vertices, codes)

pathpatch = PathPatch(path, facecolor='None', edgecolor='green')

fig, ax = plt.subplots()
ax.add_patch(pathpatch)
ax.set_title('A compound path')

ax.dataLim.update_from_data_xy(vertices)
ax.autoscale_view()


plt.show()




********************************************
./api/mathtext_asarray.py
"""
Load a mathtext image as numpy array
"""

import numpy as np
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('image', origin='upper')

parser = mathtext.MathTextParser("Bitmap")


parser.to_png('test2.png', r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$', color='green', fontsize=14, dpi=100)


rgba1, depth1 = parser.to_rgba(r'IQ: $\sigma_i=15$', color='blue', fontsize=20, dpi=200)
rgba2, depth2 = parser.to_rgba(r'some other string', color='red', fontsize=20, dpi=200)

fig = plt.figure()
fig.figimage(rgba1.astype(float)/255., 100, 100)
fig.figimage(rgba2.astype(float)/255., 100, 300)

plt.show()




********************************************
./api/sankey_demo_rankine.py
"""Demonstrate the Sankey class with a practicle example of a Rankine power cycle.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.sankey import Sankey

fig = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Rankine Power Cycle: Example 8.6 from Moran and Shapiro\n"
                           + "\x22Fundamentals of Engineering Thermodynamics\x22, 6th ed., 2008")
Hdot = [260.431, 35.078, 180.794, 221.115, 22.700,
        142.361, 10.193, 10.210, 43.670, 44.312,
        68.631, 10.758, 10.758, 0.017, 0.642,
        232.121, 44.559, 100.613, 132.168] # MW
sankey = Sankey(ax=ax, format='%.3G', unit=' MW', gap=0.5, scale=1.0/Hdot[0])
sankey.add(patchlabel='\n\nPump 1', rotation=90, facecolor='#37c959',
           flows=[Hdot[13], Hdot[6], -Hdot[7]],
           labels=['Shaft power', '', None],
           pathlengths=[0.4, 0.883, 0.25],
           orientations=[1, -1, 0])
sankey.add(patchlabel='\n\nOpen\nheater', facecolor='#37c959',
           flows=[Hdot[11], Hdot[7], Hdot[4], -Hdot[8]],
           labels=[None, '', None, None],
           pathlengths=[0.25, 0.25, 1.93, 0.25],
           orientations=[1, 0, -1, 0], prior=0, connect=(2, 1))
sankey.add(patchlabel='\n\nPump 2', facecolor='#37c959',
           flows=[Hdot[14], Hdot[8], -Hdot[9]],
           labels=['Shaft power', '', None],
           pathlengths=[0.4, 0.25, 0.25],
           orientations=[1, 0, 0], prior=1, connect=(3, 1))
sankey.add(patchlabel='Closed\nheater', trunklength=2.914, fc='#37c959',
           flows=[Hdot[9], Hdot[1], -Hdot[11], -Hdot[10]],
           pathlengths=[0.25, 1.543, 0.25, 0.25],
           labels=['', '', None, None],
           orientations=[0, -1, 1, -1], prior=2, connect=(2, 0))
sankey.add(patchlabel='Trap', facecolor='#37c959', trunklength=5.102,
           flows=[Hdot[11], -Hdot[12]],
           labels=['\n', None],
           pathlengths=[1.0, 1.01],
           orientations=[1, 1], prior=3, connect=(2, 0))
sankey.add(patchlabel='Steam\ngenerator', facecolor='#ff5555',
           flows=[Hdot[15], Hdot[10], Hdot[2], -Hdot[3], -Hdot[0]],
           labels=['Heat rate', '', '', None, None],
           pathlengths=0.25,
           orientations=[1, 0, -1, -1, -1], prior=3, connect=(3, 1))
sankey.add(patchlabel='\n\n\nTurbine 1', facecolor='#37c959',
           flows=[Hdot[0], -Hdot[16], -Hdot[1], -Hdot[2]],
           labels=['', None, None, None],
           pathlengths=[0.25, 0.153, 1.543, 0.25],
           orientations=[0, 1, -1, -1], prior=5, connect=(4, 0))
sankey.add(patchlabel='\n\n\nReheat', facecolor='#37c959',
           flows=[Hdot[2], -Hdot[2]],
           labels=[None, None],
           pathlengths=[0.725, 0.25],
           orientations=[-1, 0], prior=6, connect=(3, 0))
sankey.add(patchlabel='Turbine 2', trunklength=3.212, facecolor='#37c959',
           flows=[Hdot[3], Hdot[16], -Hdot[5], -Hdot[4], -Hdot[17]],
           labels=[None, 'Shaft power', None, '', 'Shaft power'],
           pathlengths=[0.751, 0.15, 0.25, 1.93, 0.25],
           orientations=[0, -1, 0, -1, 1], prior=6, connect=(1, 1))
sankey.add(patchlabel='Condenser', facecolor='#58b1fa', trunklength=1.764,
           flows=[Hdot[5], -Hdot[18], -Hdot[6]],
           labels=['', 'Heat rate', None],
           pathlengths=[0.45, 0.25, 0.883],
           orientations=[-1, 1, 0], prior=8, connect=(2, 0))
diagrams = sankey.finish()
for diagram in diagrams:
    diagram.text.set_fontweight('bold')
    diagram.text.set_fontsize('10')
    for text in diagram.texts:
        text.set_fontsize('10')
# Notice that the explicit connections are handled automatically, but the
# implicit ones currently are not.  The lengths of the paths and the trunks
# must be adjusted manually, and that is a bit tricky.

plt.show()




********************************************
./api/joinstyle.py
#!/usr/bin/env python
"""
Illustrate the three different join styles
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_angle(ax, x, y, angle, style):
    phi = angle/180*np.pi
    xx = [x+.5,x,x+.5*np.cos(phi)]
    yy = [y,y,y+.5*np.sin(phi)]
    ax.plot(xx, yy, lw=8, color='blue', solid_joinstyle=style)
    ax.plot(xx[1:], yy[1:], lw=1, color='black')
    ax.plot(xx[1::-1], yy[1::-1], lw=1, color='black')
    ax.plot(xx[1:2], yy[1:2], 'o', color='red', markersize=3)
    ax.text(x,y+.2,'%.0f degrees' % angle)

fig, ax = plt.subplots()
ax.set_title('Join style')

for x,style in enumerate((('miter', 'round', 'bevel'))):
    ax.text(x, 5, style)
    for i in range(5):
        plot_angle(ax, x, i, pow(2.0,3+i), style)

ax.set_xlim(-.5,2.75)
ax.set_ylim(-.5,5.5)
plt.show()




********************************************
./api/date_demo.py
#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks

All matplotlib date plotting is done by converting date instances into
days since the 0001-01-01 UTC.  The conversion, tick locating and
formatting is done behind the scenes so this is most transparent to
you.  The dates module provides several converter functions date2num
and num2date

"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years    = mdates.YearLocator()   # every year
months   = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column
datafile = cbook.get_sample_data('goog.npy')
r = np.load(datafile).view(np.recarray)

fig, ax = plt.subplots()
ax.plot(r.date, r.adj_close)


# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

datemin = datetime.date(r.date.min().year, 1, 1)
datemax = datetime.date(r.date.max().year+1, 1, 1)
ax.set_xlim(datemin, datemax)

# format the coords message box
def price(x): return '$%1.2f'%x
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()




********************************************
./api/radar_chart.py
"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def example_data():
    #The following data is from the Denver Aerosol Sources and Health study.
    #See  doi:10.1016/j.atmosenv.2008.12.017
    #
    #The data are pollution source profile estimates for five modeled pollution
    #sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical species.
    #The radar charts are experimented with here to see if we can nicely
    #visualize how the modeled source profiles change across four scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase speciesis present...
    data = {
        'column names':
            ['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO',
             'O3'],
        'Basecase':
            [[0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
             [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
             [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
             [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
             [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]],
        'With CO':
            [[0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00],
             [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00],
             [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00],
             [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00],
             [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00]],
        'With O3':
            [[0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03],
             [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00],
             [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00],
             [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95],
             [0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00]],
        'CO & O3':
            [[0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01],
             [0.09, 0.95, 0.02, 0.03, 0.00, 0.01, 0.13, 0.06, 0.00],
             [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00],
             [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88],
             [0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16]]}
    return data


if __name__ == '__main__':
    N = 9
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop('column names')

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    for n, title in enumerate(data.keys()):
        ax = fig.add_subplot(2, 2, n+1, projection='radar')
        plt.rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(data[title], colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    plt.subplot(2, 2, 1)
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')

    plt.figtext(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
                ha='center', color='black', weight='bold', size='large')
    plt.show()




********************************************
./api/font_file.py
# -*- noplot -*-
"""
Although it is usually not a good idea to explicitly point to a single
ttf file for a font instance, you can do so using the
font_manager.FontProperties fname argument (for a more flexible
solution, see the font_fmaily_rc.py and fonts_demo.py examples).
"""
import sys
import os
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3])

if sys.platform == 'win32':
    fpath = 'C:\\Windows\\Fonts\\Tahoma.ttf'
elif sys.platform.startswith('linux'):
    fonts = ['/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf',
      '/usr/share/fonts/truetype/ttf-liberation/LiberationSans-BoldItalic.ttf',
      '/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf',
      ]
    for fpath in fonts:
        if os.path.exists(fpath):
            break
else:
    fpath = '/Library/Fonts/Tahoma.ttf'

if os.path.exists(fpath):
    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]
    ax.set_title('this is a special font: %s' % fname, fontproperties=prop)
else:
    ax.set_title('Demo fails--cannot find a demo font')
ax.set_xlabel('This is the default font')

plt.show()





********************************************
./api/patch_collection.py
import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

resolution = 50 # the number of vertices
N = 3
x       = np.random.rand(N)
y       = np.random.rand(N)
radii   = 0.1*np.random.rand(N)
patches = []
for x1,y1,r in zip(x, y, radii):
    circle = Circle((x1,y1), r)
    patches.append(circle)

x       = np.random.rand(N)
y       = np.random.rand(N)
radii   = 0.1*np.random.rand(N)
theta1  = 360.0*np.random.rand(N)
theta2  = 360.0*np.random.rand(N)
for x1,y1,r,t1,t2 in zip(x, y, radii, theta1, theta2):
    wedge = Wedge((x1,y1), r, t1, t2)
    patches.append(wedge)

# Some limiting conditions on Wedge
patches += [
    Wedge((.3,.7), .1, 0, 360),             # Full circle
    Wedge((.7,.8), .2, 0, 360, width=0.05), # Full ring
    Wedge((.8,.3), .2, 0, 45),              # Full sector
    Wedge((.8,.3), .2, 45, 90, width=0.10), # Ring sector
]

for i in range(N):
    polygon = Polygon(np.random.rand(N,2), True)
    patches.append(polygon)

colors = 100*np.random.rand(len(patches))
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
plt.colorbar(p)

plt.show()




********************************************
./api/span_regions.py
"""
Illustrate some helper functions for shading regions where a logical
mask is True

See :meth:`matplotlib.collections.BrokenBarHCollection.span_where`
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections


t = np.arange(0.0, 2, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = 1.2*np.sin(4*np.pi*t)


fig, ax = plt.subplots()
ax.set_title('using span_where')
ax.plot(t, s1, color='black')
ax.axhline(0, color='black', lw=2)

collection = collections.BrokenBarHCollection.span_where(
       t, ymin=0, ymax=1, where=s1>0, facecolor='green', alpha=0.5)
ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
       t, ymin=-1, ymax=0, where=s1<0, facecolor='red', alpha=0.5)
ax.add_collection(collection)



plt.show()









********************************************
./api/watermark_image.py
"""
Use a PNG file as a watermark
"""
from __future__ import print_function
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt

datafile = cbook.get_sample_data('logo2.png', asfileobj=False)
print ('loading %s' % datafile)
im = image.imread(datafile)
im[:,:,-1] = 0.5  # set the alpha channel

fig, ax = plt.subplots()

ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()
fig.figimage(im, 10, 10)

plt.show()




********************************************
./api/barchart_demo.py

#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()




********************************************
./api/bbox_intersect.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.path import Path

rect = plt.Rectangle((-1, -1), 2, 2, facecolor="#aaaaaa")
plt.gca().add_patch(rect)
bbox = Bbox.from_bounds(-1, -1, 2, 2)

for i in range(12):
    vertices = (np.random.random((2, 2)) - 0.5) * 6.0
    path = Path(vertices)
    if path.intersects_bbox(bbox):
        color = 'r'
    else:
        color = 'b'
    plt.plot(vertices[:,0], vertices[:,1], color=color)

plt.show()




********************************************
./api/unicode_minus.py
"""
You can use the proper typesetting unicode minus (see
http://en.wikipedia.org/wiki/Plus_sign#Plus_sign) or the ASCII hypen
for minus, which some people prefer.  The matplotlibrc param
axes.unicode_minus controls the default behavior.

The default is to use the unicode minus
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o')
ax.set_title('Using hypen instead of unicode minus')
plt.show()




********************************************
./api/collections_demo.py
#!/usr/bin/env python
'''Demonstration of LineCollection, PolyCollection, and
RegularPolyCollection with autoscaling.

For the first two subplots, we will use spirals.  Their
size will be set in plot units, not data units.  Their positions
will be set in data units by using the "offsets" and "transOffset"
kwargs of the LineCollection and PolyCollection.

The third subplot will make regular polygons, with the same
type of scaling and positioning as in the first two.

The last subplot illustrates the use of "offsets=(xo,yo)",
that is, a single tuple instead of a list of tuples, to generate
successively offset curves, with the offset given in data
units.  This behavior is available only for the LineCollection.

'''

import matplotlib.pyplot as plt
from matplotlib import collections, transforms
from matplotlib.colors import colorConverter
import numpy as np

nverts = 50
npts = 100

# Make some spirals
r = np.array(range(nverts))
theta = np.array(range(nverts)) * (2*np.pi)/(nverts-1)
xx = r * np.sin(theta)
yy = r * np.cos(theta)
spiral = list(zip(xx,yy))

# Make some offsets
rs = np.random.RandomState([12345678])
xo = rs.randn(npts)
yo = rs.randn(npts)
xyo = list(zip(xo, yo))

# Make a list of colors cycling through the rgbcmyk series.
colors = [colorConverter.to_rgba(c) for c in ('r','g','b','c','y','m','k')]

fig, axes = plt.subplots(2,2)
((ax1, ax2), (ax3, ax4)) = axes # unpack the axes


col = collections.LineCollection([spiral], offsets=xyo,
                                transOffset=ax1.transData)
trans = fig.dpi_scale_trans + transforms.Affine2D().scale(1.0/72.0)
col.set_transform(trans)  # the points to pixels transform
    # Note: the first argument to the collection initializer
    # must be a list of sequences of x,y tuples; we have only
    # one sequence, but we still have to put it in a list.
ax1.add_collection(col, autolim=True)
    # autolim=True enables autoscaling.  For collections with
    # offsets like this, it is neither efficient nor accurate,
    # but it is good enough to generate a plot that you can use
    # as a starting point.  If you know beforehand the range of
    # x and y that you want to show, it is better to set them
    # explicitly, leave out the autolim kwarg (or set it to False),
    # and omit the 'ax1.autoscale_view()' call below.

# Make a transform for the line segments such that their size is
# given in points:
col.set_color(colors)

ax1.autoscale_view()  # See comment above, after ax1.add_collection.
ax1.set_title('LineCollection using offsets')


# The same data as above, but fill the curves.
col = collections.PolyCollection([spiral], offsets=xyo,
                                transOffset=ax2.transData)
trans = transforms.Affine2D().scale(fig.dpi/72.0)
col.set_transform(trans)  # the points to pixels transform
ax2.add_collection(col, autolim=True)
col.set_color(colors)


ax2.autoscale_view()
ax2.set_title('PolyCollection using offsets')

# 7-sided regular polygons

col = collections.RegularPolyCollection(7,
                                        sizes = np.fabs(xx)*10.0, offsets=xyo,
                                        transOffset=ax3.transData)
trans = transforms.Affine2D().scale(fig.dpi/72.0)
col.set_transform(trans)  # the points to pixels transform
ax3.add_collection(col, autolim=True)
col.set_color(colors)
ax3.autoscale_view()
ax3.set_title('RegularPolyCollection using offsets')


# Simulate a series of ocean current profiles, successively
# offset by 0.1 m/s so that they form what is sometimes called
# a "waterfall" plot or a "stagger" plot.

nverts = 60
ncurves = 20
offs = (0.1, 0.0)

yy = np.linspace(0, 2*np.pi, nverts)
ym = np.amax(yy)
xx = (0.2 + (ym-yy)/ym)**2 * np.cos(yy-0.4) * 0.5
segs = []
for i in range(ncurves):
    xxx = xx + 0.02*rs.randn(nverts)
    curve = list(zip(xxx, yy*100))
    segs.append(curve)

col = collections.LineCollection(segs, offsets=offs)
ax4.add_collection(col, autolim=True)
col.set_color(colors)
ax4.autoscale_view()
ax4.set_title('Successive data offsets')
ax4.set_xlabel('Zonal velocity component (m/s)')
ax4.set_ylabel('Depth (m)')
# Reverse the y-axis so depth increases downward
ax4.set_ylim(ax4.get_ylim()[::-1])


plt.show()






********************************************
./api/engineering_formatter.py
'''
Demo to show use of the engineering Formatter.
'''

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import EngFormatter

fig, ax = plt.subplots()
ax.set_xscale('log')
formatter = EngFormatter(unit='Hz', places=1)
ax.xaxis.set_major_formatter(formatter)

xs = np.logspace(1, 9, 100)
ys = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(xs)**2
ax.plot(xs, ys)

plt.show()




********************************************
./api/demo_affine_image.py
#!/usr/bin/env python


"""
For the backends that supports draw_image with optional affine
transform (e.g., agg, ps backend), the image of the output should
have its boundary matches the red rectangles.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def get_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2-Z1  # difference of Gaussians
    return Z

def imshow_affine(ax, z, *kl, **kwargs):
    im = ax.imshow(z, *kl, **kwargs)
    x1, x2, y1, y2 = im.get_extent()
    im._image_skew_coordinate = (x2, y1)
    return im


if 1:

    # image rotation

    fig, (ax1, ax2) = plt.subplots(1,2)
    Z = get_image()
    im1 = imshow_affine(ax1, Z, interpolation='none', cmap=cm.jet,
                        origin='lower',
                        extent=[-2, 4, -3, 2], clip_on=True)

    trans_data2 = mtransforms.Affine2D().rotate_deg(30) + ax1.transData
    im1.set_transform(trans_data2)

    # display intended extent of the image
    x1, x2, y1, y2 = im1.get_extent()
    x3, y3 = x2, y1

    ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r--", lw=3,
             transform=trans_data2)

    ax1.set_xlim(-3, 5)
    ax1.set_ylim(-4, 4)


    # image skew

    im2 = ax2.imshow(Z, interpolation='none', cmap=cm.jet,
                     origin='lower',
                     extent=[-2, 4, -3, 2], clip_on=True)
    im2._image_skew_coordinate = (3, -2)


    plt.show()
    #plt.savefig("demo_affine_image")




********************************************
./api/sankey_demo_links.py
"""Demonstrate/test the Sankey class by producing a long chain of connections.
"""

from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

links_per_side = 6


def side(sankey, n=1):
    """Generate a side chain."""
    prior = len(sankey.diagrams)
    colors = cycle(['orange', 'b', 'g', 'r', 'c', 'm', 'y'])
    for i in range(0, 2*n, 2):
        sankey.add(flows=[1, -1], orientations=[-1, -1],
                   patchlabel=str(prior+i), facecolor=next(colors),
                   prior=prior+i-1, connect=(1, 0), alpha=0.5)
        sankey.add(flows=[1, -1], orientations=[1, 1],
                   patchlabel=str(prior+i+1), facecolor=next(colors),
                   prior=prior+i, connect=(1, 0), alpha=0.5)


def corner(sankey):
    """Generate a corner link."""
    prior = len(sankey.diagrams)
    sankey.add(flows=[1, -1], orientations=[0, 1],
               patchlabel=str(prior), facecolor='k',
               prior=prior-1, connect=(1, 0), alpha=0.5)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Why would you want to do this?\n(But you could.)")
sankey = Sankey(ax=ax, unit=None)
sankey.add(flows=[1, -1], orientations=[0, 1],
           patchlabel="0", facecolor='k',
           rotation=45)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
sankey.finish()
# Notice:
# 1. The alignment doesn't drift significantly (if at all; with 16007
#    subdiagrams there is still closure).
# 2. The first diagram is rotated 45 deg, so all other diagrams are rotated
#    accordingly.

plt.show()




********************************************
./api/custom_projection_example.py
from __future__ import unicode_literals

import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis

import numpy as np

# This example projection class is rather long, but it is designed to
# illustrate many features, not all of which will be used every time.
# It is also common to factor out a lot of these methods into common
# code used by a number of projections with similar characteristics
# (see geo.py).

class HammerAxes(Axes):
    """
    A custom class for the Aitoff-Hammer projection, an equal-area map
    projection.

    http://en.wikipedia.org/wiki/Hammer_projection
    """
    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='custom_hammer')``.
    name = 'custom_hammer'

    def __init__(self, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        # Do not register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until HammerAxes.xaxis.cla() works.
        # self.spines['hammer'].register_axis(self.yaxis)
        self._update_transScale()

    def cla(self):
        """
        Override to set up some reasonable defaults.
        """
        # Don't forget to call the base class
        Axes.cla(self)

        # Set up a default grid spacing
        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)

        # Turn off minor ticking altogether
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())

        # Do not display ticks -- we only want gridlines and text
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')

        # The limits on this projection are fixed -- they are not to
        # be changed by the user.  This makes the math in the
        # transformation itself easier, and since this is a toy
        # example, the easier, the better.
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # There are three important coordinate spaces going on here:
        #
        #    1. Data space: The space of the data itself
        #
        #    2. Axes space: The unit rectangle (0, 0) to (1, 1)
        #       covering the entire plot area.
        #
        #    3. Display space: The coordinates of the resulting image,
        #       often in pixels or dpi/inch.

        # This function makes heavy use of the Transform classes in
        # ``lib/matplotlib/transforms.py.`` For more information, see
        # the inline documentation there.

        # The goal of the first two transformations is to get from the
        # data space (in this case longitude and latitude) to axes
        # space.  It is separated into a non-affine and affine part so
        # that the non-affine part does not have to be recomputed when
        # a simple affine change to the figure has been made (such as
        # resizing the window or changing the dpi).

        # 1) The core transformation from data space into
        # rectilinear space defined in the HammerTransform class.
        self.transProjection = self.HammerTransform()

        # 2) The above has an output range that is not in the unit
        # rectangle, so scale and translate it so it fits correctly
        # within the axes.  The peculiar calculations of xscale and
        # yscale are specific to a Aitoff-Hammer projection, so don't
        # worry about them too much.
        xscale = 2.0 * np.sqrt(2.0) * np.sin(0.5 * np.pi)
        yscale = np.sqrt(2.0) * np.sin(0.5 * np.pi)
        self.transAffine = Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

        # 3) This is the transformation from axes space to display
        # space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Now put these 3 transforms together -- from data all the way
        # to display coordinates.  Using the '+' operator, these
        # transforms will be applied "in order".  The transforms are
        # automatically simplified, if possible, by the underlying
        # transformation framework.
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        # The main data transformation is set up.  Now deal with
        # gridlines and tick labels.

        # Longitude gridlines and ticklabels.  The input to these
        # transforms are in display space in x and axes space in y.
        # Therefore, the input values will be in range (-xmin, 0),
        # (xmax, 1).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the equator.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, np.pi) \
            .translate(0.0, -np.pi)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, 4.0)
        self._xaxis_text2_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -4.0)

        # Now set up the transforms for the latitude ticks.  The input to
        # these transforms are in axes space in x and display space in
        # y.  Therefore, the input values will be in range (0, -ymin),
        # (1, ymax).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the edge of the axes ellipse.
        yaxis_stretch = Affine2D().scale(np.pi * 2.0, 1.0).translate(-np.pi, 0.0)
        yaxis_space = Affine2D().scale(1.0, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space + \
             self.transAffine + \
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8.0, 0.0)

    def get_xaxis_transform(self,which='grid'):
        """
        Override this method to provide a transformation for the
        x-axis grid and ticks.
        """
        assert which in ['tick1','tick2','grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        secondary x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self,which='grid'):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        assert which in ['tick1','tick2','grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        secondary y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        return {'custom_hammer':mspines.Spine.circular_spine(self,
                                                      (0.5, 0.5), 0.5)}

    # Prevent the user from applying scales to one or both of the
    # axes.  In this particular case, scaling the axes wouldn't make
    # sense, so we don't allow it.
    def set_xscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_xscale(self, *args, **kwargs)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_yscale(self, *args, **kwargs)

    # Prevent the user from changing the axes limits.  In our case, we
    # want to display the whole sphere all the time, so we override
    # set_xlim and set_ylim to ignore any input.  This also applies to
    # interactive panning and zooming in the GUI interfaces.
    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)
    set_ylim = set_xlim

    def format_coord(self, lon, lat):
        """
        Override this method to change how the values are displayed in
        the status bar.

        In this case, we want them to be displayed in degrees N/S/E/W.
        """
        lon = lon * (180.0 / np.pi)
        lat = lat * (180.0 / np.pi)
        if lat >= 0.0:
            ns = 'N'
        else:
            ns = 'S'
        if lon >= 0.0:
            ew = 'E'
        else:
            ew = 'W'
        # \u00b0 : degree symbol
        return '%f\u00b0%s, %f\u00b0%s' % (abs(lat), ns, abs(lon), ew)

    class DegreeFormatter(Formatter):
        """
        This is a custom formatter that converts the native unit of
        radians into (truncated) degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = (x / np.pi) * 180.0
            degrees = round(degrees / self._round_to) * self._round_to
            # \u00b0 : degree symbol
            return "%d\u00b0" % degrees

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface to set the
        ticking than set_xticks would.
        """
        # Set up a FixedLocator at each of the points, evenly spaced
        # by degrees.
        number = (360.0 / degrees) + 1
        self.xaxis.set_major_locator(
            plt.FixedLocator(
                np.linspace(-np.pi, np.pi, number, True)[1:-1]))
        # Set the formatter to display the tick labels in degrees,
        # rather than radians.
        self.xaxis.set_major_formatter(self.DegreeFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface than
        set_yticks would.
        """
        # Set up a FixedLocator at each of the points, evenly spaced
        # by degrees.
        number = (180.0 / degrees) + 1
        self.yaxis.set_major_locator(
            FixedLocator(
                np.linspace(-np.pi / 2.0, np.pi / 2.0, number, True)[1:-1]))
        # Set the formatter to display the tick labels in degrees,
        # rather than radians.
        self.yaxis.set_major_formatter(self.DegreeFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.

        Often, in geographic projections, you wouldn't want to draw
        longitude gridlines near the poles.  This allows the user to
        specify the degree at which to stop drawing longitude grids.

        This is an example method that is specific to this projection
        class -- it provides an interface to something that has no
        analogy in the base Axes class.
        """
        longitude_cap = degrees * (np.pi / 180.0)
        # Change the xaxis gridlines transform so that it draws from
        # -degrees to degrees, rather than -pi to pi.
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, longitude_cap * 2.0) \
            .translate(0.0, -longitude_cap)

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.

        This method should be overridden by any Axes that have a
        fixed data ratio.
        """
        return 1.0

    # Interactive panning and zooming is not supported with this projection,
    # so we override all of the following methods to disable it.
    def can_zoom(self):
        """
        Return True if this axes support the zoom box
        """
        return False
    def start_pan(self, x, y, button):
        pass
    def end_pan(self):
        pass
    def drag_pan(self, button, key, x, y):
        pass

    # Now, the transforms themselves.

    class HammerTransform(Transform):
        """
        The base Hammer transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform_non_affine(self, ll):
            """
            Override the transform_non_affine method to implement the custom 
            transform.

            The input and output are Nx2 numpy arrays.
            """
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]

            # Pre-compute some values
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)
            sqrt2 = np.sqrt(2.0)

            alpha = 1.0 + cos_latitude * np.cos(half_long)
            x = (2.0 * sqrt2) * (cos_latitude * np.sin(half_long)) / alpha
            y = (sqrt2 * np.sin(latitude)) / alpha
            return np.concatenate((x, y), 1)

        # This is where things get interesting.  With this projection,
        # straight lines in data space become curves in display space.
        # This is done by interpolating new values between the input
        # values of the data.  Since ``transform`` must not return a
        # differently-sized array, any transform that requires
        # changing the length of the data array must happen within
        # ``transform_path``.
        def transform_path_non_affine(self, path):
            ipath = path.interpolated(path._interpolation_steps)
            return Path(self.transform(ipath.vertices), ipath.codes)
        transform_path_non_affine.__doc__ = \
                Transform.transform_path_non_affine.__doc__

        if matplotlib.__version__ < '1.2':
            # Note: For compatibility with matplotlib v1.1 and older, you'll
            # need to explicitly implement a ``transform`` method as well.
            # Otherwise a ``NotImplementedError`` will be raised. This isn't
            # necessary for v1.2 and newer, however.  
            transform = transform_non_affine

            # Similarly, we need to explicitly override ``transform_path`` if
            # compatibility with older matplotlib versions is needed. With v1.2
            # and newer, only overriding the ``transform_path_non_affine``
            # method is sufficient.
            transform_path = transform_path_non_affine
            transform_path.__doc__ = Transform.transform_path.__doc__

        def inverted(self):
            return HammerAxes.InvertedHammerTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedHammerTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform_non_affine(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]

            quarter_x = 0.25 * x
            half_y = 0.5 * y
            z = np.sqrt(1.0 - quarter_x*quarter_x - half_y*half_y)
            longitude = 2 * np.arctan((z*x) / (2.0 * (2.0*z*z - 1.0)))
            latitude = np.arcsin(y*z)
            return np.concatenate((longitude, latitude), 1)
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        # As before, we need to implement the "transform" method for 
        # compatibility with matplotlib v1.1 and older.
        if matplotlib.__version__ < '1.2':
            transform = transform_non_affine

        def inverted(self):
            # The inverse of the inverse is the original transform... ;)
            return HammerAxes.HammerTransform()
        inverted.__doc__ = Transform.inverted.__doc__

# Now register the projection with matplotlib so the user can select
# it.
register_projection(HammerAxes)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Now make a simple example using the custom projection.
    plt.subplot(111, projection="custom_hammer")
    p = plt.plot([-1, 1, 1], [-1, -1, 1], "o-")
    plt.grid(True)

    plt.show()





********************************************
./api/legend_demo.py
"""
Demo of the legend function with a few features.

In addition to the basic legend, this demo shows a few optional features:

    * Custom legend placement.
    * A keyword argument to a drop-shadow.
    * Setting the background color.
    * Setting the font size.
    * Setting the line width.
"""
import numpy as np
import matplotlib.pyplot as plt


# Example data
a = np.arange(0,3, .02)
b = np.arange(0,3, .02)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c+d, 'k', label='Total message length')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame  = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()




********************************************
./api/logo2.py
"""
Thanks to Tony Yu <tsyu80@gmail.com> for the logo design
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = 'gray'


axalpha = 0.05
#figcolor = '#EFEFEF'
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(6, 1.1),dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)


def add_math_background():
    ax = fig.add_axes([0., 0., 1., 1.])

    text = []
    text.append((r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$", (0.7, 0.2), 20))
    text.append((r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} = -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$",
                (0.35, 0.9), 20))
    text.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$",
                (0.15, 0.3), 25))
    #text.append((r"$E = mc^2 = \sqrt{{m_0}^2c^4 + p^2c^2}$",
    #            (0.7, 0.42), 30))
    text.append((r"$F_G = G\frac{m_1m_2}{r^2}$",
                (0.85, 0.7), 30))
    for eq, (x, y), size in text:
        ax.text(x, y, eq, ha='center', va='center', color="#11557c", alpha=0.25,
                transform=ax.transAxes, fontsize=size)
    ax.set_axis_off()
    return ax

def add_matplotlib_text(ax):
    ax.text(0.95, 0.5, 'matplotlib', color='#11557c', fontsize=65,
               ha='right', va='center', alpha=1.0, transform=ax.transAxes)

def add_polar_bar():
    ax = fig.add_axes([0.025, 0.075, 0.2, 0.85], polar=True)


    ax.axesPatch.set_alpha(axalpha)
    ax.set_axisbelow(True)
    N = 7
    arc = 2. * np.pi
    theta = np.arange(0.0, arc, arc/N)
    radii = 10 * np.array([0.2, 0.6, 0.8, 0.7, 0.4, 0.5, 0.8])
    width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(cm.jet(r/10.))
        bar.set_alpha(0.6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(False)

    for line in ax.get_ygridlines() + ax.get_xgridlines():
        line.set_lw(0.8)
        line.set_alpha(0.9)
        line.set_ls('-')
        line.set_color('0.5')

    ax.set_yticks(np.arange(1, 9, 2))
    ax.set_rmax(9)

if __name__ == '__main__':
    main_axes = add_math_background()
    add_polar_bar()
    add_matplotlib_text(main_axes)
    plt.show()




********************************************
./api/image_zcoord.py
"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = 10*np.random.rand(5,3)

fig, ax = plt.subplots()
ax.imshow(X, cmap=cm.jet, interpolation='nearest')

numrows, numcols = X.shape
def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

ax.format_coord = format_coord
plt.show()





********************************************
./api/font_family_rc.py
"""
You can explicitly set which font family is picked up for a given font
style (eg 'serif', 'sans-serif', or 'monospace').

In the example below, we only allow one font family (Tahoma) for the
san-serif font style.  You the default family with the font.family rc
param, eg::

  rcParams['font.family'] = 'sans-serif'

and for the font.family you set a list of font styles to try to find
in order::

  rcParams['font.sans-serif'] = ['Tahoma', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana']

"""

# -*- noplot -*-

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3], label='test')

ax.legend()
plt.show()





********************************************
./images_contours_and_fields/streamplot_demo_masking.py
"""
Demo of the streamplot function with masking.

This example shows how streamlines created by the streamplot function skips
masked regions and NaN values.
"""
import numpy as np
import matplotlib.pyplot as plt

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

mask = np.zeros(U.shape, dtype=bool)
mask[40:60, 40:60] = 1
U = np.ma.array(U, mask=mask)
U[:20, :20] = np.nan

plt.streamplot(X, Y, U, V, color='r')

plt.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
           interpolation='nearest', cmap=plt.cm.gray)

plt.show()





********************************************
./images_contours_and_fields/pcolormesh_levels.py
"""
Shows how to combine Normalization and Colormap instances to draw
"levels" in pcolor, pcolormesh and imshow type plots in a similar
way to the levels keyword argument to contour/contourf.

"""

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(1, 5 + dy, dy),
                slice(1, 5 + dx, dx)]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.subplot(2, 1, 1)
im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.title('pcolormesh with levels')



plt.subplot(2, 1, 2)
# contours are *point* based plots, so convert our bound into point
# centers
plt.contourf(x[:-1, :-1] + dx / 2.,
             y[:-1, :-1] + dy / 2., z, levels=levels,
             cmap=cmap)
plt.colorbar()
plt.title('contourf with levels')


plt.show()



********************************************
./images_contours_and_fields/image_demo_clip_path.py
"""
Demo of image that's been clipped by a circular patch.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook


image_file = cbook.get_sample_data('grace_hopper.png')
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
patch = patches.Circle((260, 200), radius=200, transform=ax.transData)
im.set_clip_path(patch)

plt.axis('off')
plt.show()




********************************************
./images_contours_and_fields/streamplot_demo_features.py
"""
Demo of the `streamplot` function.

A streamplot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the stream plot function:

    * Varying the color along a streamline.
    * Varying the density of streamlines.
    * Varying the line width along a stream line.
"""
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

plt.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
plt.colorbar()

f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

lw = 5*speed/speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()





********************************************
./images_contours_and_fields/image_demo.py
"""
Simple demo of the imshow function.
"""
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

image_file = cbook.get_sample_data('ada.png')
image = plt.imread(image_file)

plt.imshow(image)
plt.axis('off') # clear x- and y-axes
plt.show()





********************************************
./lines_bars_and_markers/fill_demo.py
"""
Simple demo of the fill function.
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

plt.fill(x, y, 'r')
plt.grid(True)
plt.show()




********************************************
./lines_bars_and_markers/fill_demo_features.py
"""
Demo of the fill function with a few features.

In addition to the basic fill plot, this demo shows a few optional features:

    * Multiple curves with a single command.
    * Setting the fill color.
    * Setting the opacity (alpha value).
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(3 * x)
plt.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)
plt.show()




********************************************
./lines_bars_and_markers/line_demo_dash_control.py
"""
Demo of a simple plot with a custom dashed line.

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10)
line, = plt.plot(x, np.sin(x), '--', linewidth=2)

dashes = [10, 5, 100, 5] # 10 points on, 5 off, 100 on, 5 off
line.set_dashes(dashes)

plt.show()




********************************************
./lines_bars_and_markers/barh_demo.py
"""
Simple demo of a horizontal bar chart.
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel('Performance')
plt.title('How fast do you want to go today?')

plt.show()




********************************************
./specialty_plots/hinton_demo.py
"""
Demo of a function to create Hinton diagrams.

Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
a weight matrix): Positive and negative values are represented by white and
black squares, respectively, and the size of each square represents the
magnitude of each value.

Initial idea from David Warde-Farley on the SciPy Cookbook
"""
import numpy as np
import matplotlib.pyplot as plt


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


if __name__ == '__main__':
    hinton(np.random.rand(20, 20) - 0.5)
    plt.show()





********************************************
./pie_and_polar_charts/polar_scatter_demo.py
"""
Demo of scatter plot on a polar axis.

Size increases radially in this example and color increases with angle (just to
verify the symbols are being scattered correctly).
"""
import numpy as np
import matplotlib.pyplot as plt


N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2 * np.random.rand(N)
colors = theta

ax = plt.subplot(111, polar=True)
c = plt.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
c.set_alpha(0.75)

plt.show()




********************************************
./pie_and_polar_charts/polar_bar_demo.py
"""
Demo of bar plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.5)

plt.show()




********************************************
./pie_and_polar_charts/pie_demo_features.py
"""
Demo of a basic pie chart plus a few additional features.

In addition to the basic pie chart, this demo shows a few optional features:

    * slice labels
    * auto-labeling the percentage
    * offsetting a slice with "explode"
    * drop-shadow
    * custom start angle

Note about the custom start angle:

The default ``startangle`` is 0, which would start the "Frogs" slice on the
positive x-axis. This example sets ``startangle = 90`` such that everything is
rotated counter-clockwise by 90 degrees, and the frog slice starts on the
positive y-axis.
"""
import matplotlib.pyplot as plt


# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

plt.show()




********************************************
./units/bar_demo2.py
"""
plot using a variety of cm vs inches conversions.  The example shows
how default unit instrospection works (ax1), how various keywords can
be used to set the x and y units to override the defaults (ax2, ax3,
ax4) and how one can set the xlimits using scalars (ax3, current units
assumed) or units (conversions applied to get the numbers to current
units)

"""
import numpy as np
from basic_units import cm, inch
import matplotlib.pyplot as plt


cms = cm *np.arange(0, 10, 2)
bottom=0*cm
width=0.8*cm

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.bar(cms, cms, bottom=bottom)

ax2 = fig.add_subplot(2,2,2)
ax2.bar(cms, cms, bottom=bottom, width=width, xunits=cm, yunits=inch)

ax3 = fig.add_subplot(2,2,3)
ax3.bar(cms, cms, bottom=bottom, width=width, xunits=inch, yunits=cm)
ax3.set_xlim(2, 6)  # scalars are interpreted in current units

ax4 = fig.add_subplot(2,2,4)
ax4.bar(cms, cms, bottom=bottom, width=width, xunits=inch, yunits=inch)
#fig.savefig('simple_conversion_plot.png')
ax4.set_xlim(2*cm, 6*cm) # cm are converted to inches

plt.show()




********************************************
./units/units_sample.py
"""
plot using a variety of cm vs inches conversions.  The example shows
how default unit instrospection works (ax1), how various keywords can
be used to set the x and y units to override the defaults (ax2, ax3,
ax4) and how one can set the xlimits using scalars (ax3, current units
assumed) or units (conversions applied to get the numbers to current
units)

"""
from basic_units import cm, inch
import matplotlib.pyplot as plt
import numpy

cms = cm *numpy.arange(0, 10, 2)

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(cms, cms)

ax2 = fig.add_subplot(2,2,2)
ax2.plot(cms, cms, xunits=cm, yunits=inch)

ax3 = fig.add_subplot(2,2,3)
ax3.plot(cms, cms, xunits=inch, yunits=cm)
ax3.set_xlim(3, 6)  # scalars are interpreted in current units

ax4 = fig.add_subplot(2,2,4)
ax4.plot(cms, cms, xunits=inch, yunits=inch)
#fig.savefig('simple_conversion_plot.png')
ax4.set_xlim(3*cm, 6*cm) # cm are converted to inches

plt.show()




********************************************
./units/radian_demo.py
"""
Plot with radians from the basic_units mockup example package
This example shows how the unit class can determine the tick locating,
formatting and axis labeling.
"""
import numpy as np
from basic_units import radians, degrees, cos
from matplotlib.pyplot import figure, show

x = [val*radians for val in np.arange(0, 15, 0.01)]

fig = figure()
fig.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(211)
line1, = ax.plot(x, cos(x), xunits=radians)

ax = fig.add_subplot(212)
line2, = ax.plot(x, cos(x), xunits=degrees)

show()




********************************************
./units/evans_test.py
"""
A mockup "Foo" units class which supports
conversion and different tick formatting depending on the "unit".
Here the "unit" is just a scalar conversion factor, but this example shows mpl is
entirely agnostic to what kind of units client packages use
"""
from matplotlib.cbook import iterable
import matplotlib.units as units
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


class Foo:
    def __init__( self, val, unit=1.0 ):
        self.unit = unit
        self._val = val * unit

    def value( self, unit ):
        if unit is None: unit = self.unit
        return self._val / unit

class FooConverter:

    @staticmethod
    def axisinfo(unit, axis):
        'return the Foo AxisInfo'
        if unit==1.0 or unit==2.0:
            return units.AxisInfo(
                majloc = ticker.IndexLocator( 8, 0 ),
                majfmt = ticker.FormatStrFormatter("VAL: %s"),
                label='foo',
                )

        else:
            return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        convert obj using unit.  If obj is a sequence, return the
        converted sequence
        """
        if units.ConversionInterface.is_numlike(obj):
            return obj

        if iterable(obj):
            return [o.value(unit) for o in obj]
        else:
            return obj.value(unit)

    @staticmethod
    def default_units(x, axis):
        'return the default unit for x or None'
        if iterable(x):
            for thisx in x:
                return thisx.unit
        else:
            return x.unit

units.registry[Foo] = FooConverter()

# create some Foos
x = []
for val in range( 0, 50, 2 ):
    x.append( Foo( val, 1.0 ) )

# and some arbitrary y data
y = [i for i in range( len(x) ) ]


# plot specifying units
fig = plt.figure()
fig.suptitle("Custom units")
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(1,2,2)
ax.plot( x, y, 'o', xunits=2.0 )
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
ax.set_title("xunits = 2.0")


# plot without specifying units; will use the None branch for axisinfo
ax = fig.add_subplot(1,2,1)
ax.plot( x, y ) # uses default units
ax.set_title('default units')
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')

plt.show()




********************************************
./units/basic_units.py
import math

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.cbook import iterable


class ProxyDelegate(object):
    def __init__(self, fn_name, proxy_type):
        self.proxy_type = proxy_type
        self.fn_name = fn_name

    def __get__(self, obj, objtype=None):
        return self.proxy_type(self.fn_name, obj)


class TaggedValueMeta (type):
    def __init__(cls, name, bases, dict):
        for fn_name in cls._proxies.keys():
            try:
                dummy = getattr(cls, fn_name)
            except AttributeError:
                setattr(cls, fn_name,
                        ProxyDelegate(fn_name, cls._proxies[fn_name]))


class PassThroughProxy(object):
    def __init__(self, fn_name, obj):
        self.fn_name = fn_name
        self.target = obj.proxy_target

    def __call__(self, *args):
        fn = getattr(self.target, self.fn_name)
        ret = fn(*args)
        return ret


class ConvertArgsProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        converted_args = []
        for a in args:
            try:
                converted_args.append(a.convert_to(self.unit))
            except AttributeError:
                converted_args.append(TaggedValue(a, self.unit))
        converted_args = tuple([c.get_value() for c in converted_args])
        return PassThroughProxy.__call__(self, *converted_args)


class ConvertReturnProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        ret = PassThroughProxy.__call__(self, *args)
        if (type(ret) == type(NotImplemented)):
            return NotImplemented
        return TaggedValue(ret, self.unit)


class ConvertAllProxy(PassThroughProxy):
    def __init__(self, fn_name, obj):
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        converted_args = []
        arg_units = [self.unit]
        for a in args:
            if hasattr(a, 'get_unit') and not hasattr(a, 'convert_to'):
                # if this arg has a unit type but no conversion ability,
                # this operation is prohibited
                return NotImplemented

            if hasattr(a, 'convert_to'):
                try:
                    a = a.convert_to(self.unit)
                except:
                    pass
                arg_units.append(a.get_unit())
                converted_args.append(a.get_value())
            else:
                converted_args.append(a)
                if hasattr(a, 'get_unit'):
                    arg_units.append(a.get_unit())
                else:
                    arg_units.append(None)
        converted_args = tuple(converted_args)
        ret = PassThroughProxy.__call__(self, *converted_args)
        if (type(ret) == type(NotImplemented)):
            return NotImplemented
        ret_unit = unit_resolver(self.fn_name, arg_units)
        if (ret_unit == NotImplemented):
            return NotImplemented
        return TaggedValue(ret, ret_unit)


class _TaggedValue(object):

    _proxies = {'__add__': ConvertAllProxy,
                '__sub__': ConvertAllProxy,
                '__mul__': ConvertAllProxy,
                '__rmul__': ConvertAllProxy,
                '__cmp__': ConvertAllProxy,
                '__lt__': ConvertAllProxy,
                '__gt__': ConvertAllProxy,
                '__len__': PassThroughProxy}

    def __new__(cls, value, unit):
        # generate a new subclass for value
        value_class = type(value)
        try:
            subcls = type('TaggedValue_of_%s' % (value_class.__name__),
                          tuple([cls, value_class]),
                          {})
            if subcls not in units.registry:
                units.registry[subcls] = basicConverter
            return object.__new__(subcls, value, unit)
        except TypeError:
            if cls not in units.registry:
                units.registry[cls] = basicConverter
            return object.__new__(cls, value, unit)

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit
        self.proxy_target = self.value

    def  __getattribute__(self, name):
        if (name.startswith('__')):
            return object.__getattribute__(self, name)
        variable = object.__getattribute__(self, 'value')
        if (hasattr(variable, name) and name not in self.__class__.__dict__):
            return getattr(variable, name)
        return object.__getattribute__(self, name)

    def __array__(self, t=None, context=None):
        if t is not None:
            return np.asarray(self.value).astype(t)
        else:
            return np.asarray(self.value, 'O')

    def __array_wrap__(self, array, context):
        return TaggedValue(array, self.unit)

    def __repr__(self):
        return 'TaggedValue(' + repr(self.value) + ', ' + repr(self.unit) + ')'

    def __str__(self):
        return str(self.value) + ' in ' + str(self.unit)

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        class IteratorProxy(object):
            def __init__(self, iter, unit):
                self.iter = iter
                self.unit = unit

            def __next__(self):
                value = next(self.iter)
                return TaggedValue(value, self.unit)
            next = __next__  # for Python 2
        return IteratorProxy(iter(self.value), self.unit)

    def get_compressed_copy(self, mask):
        new_value = np.ma.masked_array(self.value, mask=mask).compressed()
        return TaggedValue(new_value, self.unit)

    def convert_to(self, unit):
        if (unit == self.unit or not unit):
            return self
        new_value = self.unit.convert_value_to(self.value, unit)
        return TaggedValue(new_value, unit)

    def get_value(self):
        return self.value

    def get_unit(self):
        return self.unit


TaggedValue = TaggedValueMeta('TaggedValue', (_TaggedValue, ), {})


class BasicUnit(object):
    def __init__(self, name, fullname=None):
        self.name = name
        if fullname is None:
            fullname = name
        self.fullname = fullname
        self.conversions = dict()

    def __repr__(self):
        return 'BasicUnit(%s)'%self.name

    def __str__(self):
        return self.fullname

    def __call__(self, value):
        return TaggedValue(value, self)

    def __mul__(self, rhs):
        value = rhs
        unit = self
        if hasattr(rhs, 'get_unit'):
            value = rhs.get_value()
            unit = rhs.get_unit()
            unit = unit_resolver('__mul__', (self, unit))
        if (unit == NotImplemented):
            return NotImplemented
        return TaggedValue(value, unit)

    def __rmul__(self, lhs):
        return self*lhs

    def __array_wrap__(self, array, context):
        return TaggedValue(array, self)

    def __array__(self, t=None, context=None):
        ret = np.array([1])
        if t is not None:
            return ret.astype(t)
        else:
            return ret

    def add_conversion_factor(self, unit, factor):
        def convert(x):
            return x*factor
        self.conversions[unit] = convert

    def add_conversion_fn(self, unit, fn):
        self.conversions[unit] = fn

    def get_conversion_fn(self, unit):
        return self.conversions[unit]

    def convert_value_to(self, value, unit):
        conversion_fn = self.conversions[unit]
        ret = conversion_fn(value)
        return ret

    def get_unit(self):
        return self


class UnitResolver(object):
    def addition_rule(self, units):
        for unit_1, unit_2 in zip(units[:-1], units[1:]):
            if (unit_1 != unit_2):
                return NotImplemented
        return units[0]

    def multiplication_rule(self, units):
        non_null = [u for u in units if u]
        if (len(non_null) > 1):
            return NotImplemented
        return non_null[0]

    op_dict = {
        '__mul__': multiplication_rule,
        '__rmul__': multiplication_rule,
        '__add__': addition_rule,
        '__radd__': addition_rule,
        '__sub__': addition_rule,
        '__rsub__': addition_rule}

    def __call__(self, operation, units):
        if (operation not in self.op_dict):
            return NotImplemented

        return self.op_dict[operation](self, units)


unit_resolver = UnitResolver()

cm = BasicUnit('cm', 'centimeters')
inch = BasicUnit('inch', 'inches')
inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1/2.54)

radians = BasicUnit('rad', 'radians')
degrees = BasicUnit('deg', 'degrees')
radians.add_conversion_factor(degrees, 180.0/np.pi)
degrees.add_conversion_factor(radians, np.pi/180.0)

secs = BasicUnit('s', 'seconds')
hertz = BasicUnit('Hz', 'Hertz')
minutes = BasicUnit('min', 'minutes')

secs.add_conversion_fn(hertz, lambda x: 1./x)
secs.add_conversion_factor(minutes, 1/60.0)


# radians formatting
def rad_fn(x, pos=None):
    n = int((x / np.pi) * 2.0 + 0.25)
    if n == 0:
        return '0'
    elif n == 1:
        return r'$\pi/2$'
    elif n == 2:
        return r'$\pi$'
    elif n % 2 == 0:
        return r'$%s\pi$' % (n/2,)
    else:
        return r'$%s\pi/2$' % (n,)


class BasicUnitConverter(units.ConversionInterface):

    @staticmethod
    def axisinfo(unit, axis):
        'return AxisInfo instance for x and unit'

        if unit==radians:
            return units.AxisInfo(
                majloc=ticker.MultipleLocator(base=np.pi/2),
                majfmt=ticker.FuncFormatter(rad_fn),
                label=unit.fullname,
            )
        elif unit==degrees:
            return units.AxisInfo(
                majloc=ticker.AutoLocator(),
                majfmt=ticker.FormatStrFormatter(r'$%i^\circ$'),
                label=unit.fullname,
            )
        elif unit is not None:
            if hasattr(unit, 'fullname'):
                return units.AxisInfo(label=unit.fullname)
            elif hasattr(unit, 'unit'):
                return units.AxisInfo(label=unit.unit.fullname)
        return None

    @staticmethod
    def convert(val, unit, axis):
        if units.ConversionInterface.is_numlike(val):
            return val
        if iterable(val):
            return [thisval.convert_to(unit).get_value() for thisval in val]
        else:
            return val.convert_to(unit).get_value()

    @staticmethod
    def default_units(x, axis):
        'return the default unit for x or None'
        if iterable(x):
            for thisx in x:
                return thisx.unit
        return x.unit


def cos(x):
    if iterable(x):
        return [math.cos(val.convert_to(radians).get_value()) for val in x]
    else:
        return math.cos(x.convert_to(radians).get_value())


basicConverter = BasicUnitConverter()
units.registry[BasicUnit] = basicConverter
units.registry[TaggedValue] = basicConverter




********************************************
./units/artist_tests.py
"""
Test unit support with each of the matplotlib primitive artist types

The axes handles unit conversions and the artists keep a pointer to
their axes parent, so you must init the artists with the axes instance
if you want to initialize them with unit data, or else they will not
know how to convert the units to scalars
"""
import random
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.text as text
import matplotlib.collections as collections

from basic_units import cm, inch
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.xaxis.set_units(cm)
ax.yaxis.set_units(cm)

if 0:
    # test a line collection
    # Not supported at present.
    verts = []
    for i in range(10):
        # a random line segment in inches
        verts.append(zip(*inch*10*np.random.rand(2, random.randint(2,15))))
    lc = collections.LineCollection(verts, axes=ax)
    ax.add_collection(lc)

# test a plain-ol-line
line = lines.Line2D([0*cm, 1.5*cm], [0*cm, 2.5*cm], lw=2, color='black', axes=ax)
ax.add_line(line)

if 0:
    # test a patch
    # Not supported at present.
    rect = patches.Rectangle( (1*cm, 1*cm), width=5*cm, height=2*cm, alpha=0.2, axes=ax)
    ax.add_patch(rect)


t = text.Text(3*cm, 2.5*cm, 'text label', ha='left', va='bottom', axes=ax)
ax.add_artist(t)

ax.set_xlim(-1*cm, 10*cm)
ax.set_ylim(-1*cm, 10*cm)
#ax.xaxis.set_units(inch)
ax.grid(True)
ax.set_title("Artists with units")
plt.show()





********************************************
./units/units_scatter.py
"""
Demonstrate unit handling

basic_units is a mockup of a true units package used for testing
purposed, which illustrates the basic interface that a units package
must provide to matplotlib.

The example below shows support for unit conversions over masked
arrays.
"""
import numpy as np
from basic_units import secs, hertz, minutes
from matplotlib.pylab import figure, show

# create masked array


xsecs = secs*np.ma.MaskedArray((1,2,3,4,5,6,7,8), (1,0,1,0,0,0,1,0), np.float)
#xsecs = secs*np.arange(1,10.)

fig = figure()
ax1 = fig.add_subplot(3,1,1)
ax1.scatter(xsecs, xsecs)
#ax1.set_ylabel('seconds')
ax1.axis([0,10,0,10])

ax2 = fig.add_subplot(3,1,2, sharex=ax1)
ax2.scatter(xsecs, xsecs, yunits=hertz)
ax2.axis([0,10,0,1])

ax3 = fig.add_subplot(3,1,3, sharex=ax1)
ax3.scatter(xsecs, xsecs, yunits=hertz)
ax3.yaxis.set_units(minutes)
ax3.axis([0,10,0,1])

show()





********************************************
./units/annotate_with_units.py
import matplotlib.pyplot as plt
from basic_units import cm

fig, ax = plt.subplots()

ax.annotate( "Note 01", [0.5*cm,  0.5*cm] )

# xy and text both unitized
ax.annotate('local max', xy=(3*cm, 1*cm),  xycoords='data',
            xytext=(0.8*cm, 0.95*cm), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

# mixing units w/ nonunits
ax.annotate('local max', xy=(3*cm, 1*cm),  xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')


ax.set_xlim(0*cm, 4*cm)
ax.set_ylim(0*cm, 4*cm)
plt.show()





********************************************
./units/ellipse_with_units.py
"""
Compare the ellipse generated with arcs versus a polygonal approximation
"""
from basic_units import cm
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


xcenter, ycenter = 0.38*cm, 0.52*cm
#xcenter, ycenter = 0., 0.
width, height = 1e-1*cm, 3e-1*cm
angle = -30

theta = np.arange(0.0, 360.0, 1.0)*np.pi/180.0
x = 0.5 * width * np.cos(theta)
y = 0.5 * height * np.sin(theta)

rtheta = angle*np.pi/180.
R = np.array([
    [np.cos(rtheta),  -np.sin(rtheta)],
    [np.sin(rtheta), np.cos(rtheta)],
    ])


x, y = np.dot(R, np.array([x, y]))
x += xcenter
y += ycenter

fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Ellipse((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Ellipse((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)

#fig.savefig('ellipse_compare.png')
fig.savefig('ellipse_compare')

fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Arc((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Arc((xcenter, ycenter), width, height,
             angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)

#fig.savefig('arc_compare.png')
fig.savefig('arc_compare')

plt.show()




********************************************
./units/bar_unit_demo.py
#!/usr/bin/env python
import numpy as np
from basic_units import cm, inch
import matplotlib.pyplot as plt


N = 5
menMeans = (150*cm, 160*cm, 146*cm, 172*cm, 155*cm)
menStd =   ( 20*cm,  30*cm,  32*cm,  10*cm,  20*cm)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, menMeans, width, color='r', bottom=0*cm, yerr=menStd)


womenMeans = (145*cm, 149*cm, 172*cm, 165*cm, 200*cm)
womenStd =   (30*cm, 25*cm, 20*cm, 31*cm, 22*cm)
p2 = ax.bar(ind+width, womenMeans, width, color='y', bottom=0*cm, yerr=womenStd)

ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

ax.legend( (p1[0], p2[0]), ('Men', 'Women') )
ax.yaxis.set_units(inch)
ax.autoscale_view()

#plt.savefig('barchart_demo')
plt.show()




********************************************
./text_labels_and_annotations/unicode_demo.py
# -*- coding: utf-8 -*-
"""
Demo of unicode support in text and labels.
"""
from __future__ import unicode_literals

import matplotlib.pyplot as plt


plt.title('Développés et fabriqués')
plt.xlabel("réactivité nous permettent d'être sélectionnés et adoptés")
plt.ylabel('André was here!')
plt.text( 0.2, 0.8, 'Institut für Festkörperphysik', rotation=45)
plt.text( 0.4, 0.2, 'AVA (check kerning)')

plt.show()




********************************************
./text_labels_and_annotations/text_demo_fontdict.py
"""
Demo using fontdict to control style of text and labels.
"""
import numpy as np
import matplotlib.pyplot as plt


font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2 * np.pi * x) * np.exp(-x)

plt.plot(x, y, 'k')
plt.title('Damped exponential decay', fontdict=font)
plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('voltage (mV)', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()




********************************************
./statistics/errorbar_demo_features.py
"""
Demo of errorbar function with different ways of specifying error bars.

Errors can be specified as a constant value (as shown in `errorbar_demo.py`),
or as demonstrated in this example, they can be specified by an N x 1 or 2 x N,
where N is the number of data points.

N x 1:
    Error varies for each point, but the error values are symmetric (i.e. the
    lower and upper values are equal).

2 x N:
    Error varies for each point, and the lower and upper limits (in that order)
    are different (asymmetric case)

In addition, this example demonstrates how to use log scale with errorbar.
"""
import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)
# example error bar values that vary with x-position
error = 0.1 + 0.2 * x
# error bar values w/ different -/+ errors
lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')

ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')
ax1.set_yscale('log')
plt.show()





********************************************
./statistics/histogram_demo_features.py
"""
Demo of the histogram (hist) function with a few features.

In addition to the basic histogram, this demo shows a few optional features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the integral of
      the histogram is 1. The resulting histogram is a probability density.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# example data
mu = 100 # mean of distribution
sigma = 15 # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()




********************************************
./statistics/errorbar_demo.py
"""
Demo of the errorbar function.
"""
import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

plt.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.show()





********************************************
./ticks_and_spines/ticklabels_demo_rotation.py
"""
Demo of custom tick-labels with user-defined rotation.
"""
import matplotlib.pyplot as plt


x = [1, 2, 3, 4]
y = [1, 4, 9, 6]
labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']

plt.plot(x, y, 'ro')
# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()




********************************************
./ticks_and_spines/spines_demo.py
"""
Basic demo of axis spines.

This demo compares a normal axes, with spines on all four sides, and an axes
with spines only on the left and bottom.
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2 * np.pi, 100)
y = 2 * np.sin(x)

fig, (ax0, ax1) = plt.subplots(nrows=2)

ax0.plot(x, y)
ax0.set_title('normal spines')

ax1.plot(x, y)
ax1.set_title('bottom-left spines')

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.5)

plt.show()




********************************************
./ticks_and_spines/spines_demo_bounds.py
"""
Demo of spines using custom bounds to limit the extent of the spine.
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
y2 = y + 0.1 * np.random.normal(size=x.shape)

fig, ax = plt.subplots()
ax.plot(x, y, 'k--')
ax.plot(x, y2, 'ro')

# set ticks and tick labels
ax.set_xlim((0, 2*np.pi))
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', '$\pi$','2$\pi$'])
ax.set_ylim((-1.5, 1.5))
ax.set_yticks([-1, 0, 1])

# Only draw spine between the y-ticks
ax.spines['left'].set_bounds(-1, 1)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()




********************************************
./ticks_and_spines/spines_demo_dropped.py
"""
Demo of spines offset from the axes (a.k.a. "dropped spines").
"""
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

image = np.random.uniform(size=(10, 10))
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()




********************************************
