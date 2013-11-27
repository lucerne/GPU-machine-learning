
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from dbscan_wiki_plot import *

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=3)

tag = np.zeros(8)

X = np.genfromtxt('total.csv', delimiter=',')


class Index:
    ind = 0
    
    def one(self, event):
        tag[0] = 1
        print "one selected"
    
    def two(self, event):
        tag[1] = 1
        print "two selected"
    
    def three(self, event):
        tag[2] = 1
        print "three selected"

    def four(self, event):
        tag[3] = 1
        print "four selected"

    def five(self, event):
        tag[4] = 1
        print "five selected"

    def six(self, event):
        tag[5] = 1
        print "six selected"

    def seven(self, event):
        tag[6] = 1
        print "seven selected"

    def eight(self, event):
        tag[7] = 1
        print "eight selected"

    def plotDBSCAN(self, event):
        D = Point(X.shape, dtype=np.float64)
        D[:] = X[:]
        dbs = DBSCAN().dbscan(D, eps, MinPts)


callback = Index()

axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
var1 = Button(axnext, 'One')
var1.on_clicked(callback.one)

axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
var2 = Button(axprev, 'Two')
var2.on_clicked(callback.two)

showbutton = plt.axes([0.6, 0.05, 0.1, 0.075])
var3 = Button(showbutton, 'Three')
var3.on_clicked(callback.three)

showbutton = plt.axes([0.5, 0.05, 0.1, 0.075])
var4 = Button(showbutton, 'Four')
var4.on_clicked(callback.four)

showbutton = plt.axes([0.4, 0.05, 0.1, 0.075])
var5 = Button(showbutton, 'Five')
var5.on_clicked(callback.five)

showbutton = plt.axes([0.3, 0.05, 0.1, 0.075])
var6 = Button(showbutton, 'Six')
var6.on_clicked(callback.six)

showbutton = plt.axes([0.2, 0.05, 0.1, 0.075])
var7 = Button(showbutton, 'Seven')
var7.on_clicked(callback.seven)

showbutton = plt.axes([0.1, 0.05, 0.1, 0.075])
var8 = Button(showbutton, 'Eight')
var8.on_clicked(callback.eight)

plotDBSCAN = plt.axes([0., 0.05, 0.1, 0.075])
plotDBSCAN = Button(plotDBSCAN, 'Plot')
plotDBSCAN.on_clicked(callback.plotDBSCAN)

callback.plotDBSCAN

plt.show()

