import sys
import math
import csv
import numpy as np
import copy 
import time
import matplotlib.pyplot as plt
# plt.pyplot.scatter(x,y)

debug = False

class Point(object):
	"""docstring for Test"""
	def __init__(self, data):
		self.data = data
		self.label = 0
		self.NOISE = False
		self.visited = False
		self.cluster = -1
		self.isClusterMember = False

class DBSCAN(object):
	"""docstring for DBSCAN"""
	def __init__(self, ):
		super(DBSCAN, self).__init__()

	def print_clusters(self, D, C):
		"""docstring for print_clusters"""
		for cluster in range(C+1):
			print "*********************"
			print "Cluster %d members are " % cluster
			for P in D:
				if P.cluster == cluster:
					print P.label
			print 


	def plot_clusters(dataX, dataY,  y,self):
		"""docstring for plot_clusters"""
		x_min   = np.min(dataX)
		x_max   = np.max(dataX)
		y_min   = np.min(dataY) 
		y_max   = np.max(dataY)
		npts    = len(dataX)
		plt.figure()
		plt.scatter(dataX, dataY)
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.title('scatter (%d points)' % npts)
		plt.show()

	
	def dbscan(self, D, eps, MinPts):
		C = -1 
		for P in D:
			assert self.in_range(P, P) == True
			if not P.visited:
				P.visited = True
				NeighborPts = self.regionQuery(P, eps)
				assert NeighborPts != None
				if len(NeighborPts) < MinPts:
					P.NOISE = True
				else:
					C += 1
					self.expandCluster(P, NeighborPts, C, eps, MinPts)
		self.print_clusters(D, C)



	def expandCluster(self, P, NeighborPts, C, eps, MinPts):
		P.cluster = C
		P.isClusterMember = True
		assert NeighborPts != None
		for P_dash in NeighborPts:
			if not P_dash.visited:
				P_dash.visited = True
				NeighborPts_dash = self.regionQuery(P_dash, eps)
				if len(NeighborPts_dash) >= MinPts:
					NeighborPts.extend(NeighborPts_dash)
				if not P_dash.isClusterMember:
					P_dash.cluster = C 
					P_dash.isClusterMember = True


	def in_range(self, point, P_dash):
		"""docstring for in_range"""
		return np.sqrt(np.sum((point.data-P_dash.data)**2)) <= eps
		
		  
	def regionQuery(self, P, eps):
		"""docstring for regionQuery"""
		return [point for point in D if self.in_range(point, P)]

def plot_clusters(dataX, dataY):
	"""docstring for plot_clusters"""
	x_min   = np.min(dataX)
	x_max   = np.max(dataX)
	y_min   = np.min(dataY) 
	y_max   = np.max(dataY)
	npts    = len(dataX)
	plt.figure()
	plt.scatter(dataX, dataY)
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.title('scatter (%d points)' % npts)
	plt.show()

			
if __name__ == '__main__':
	assert len(sys.argv) > 1, " Input file must be provided."
	input_file_name = sys.argv[1]
	X = np.genfromtxt(input_file_name, delimiter=',')
	D = []
	plot_clusters(X[:,0], X[:,1])
	sys.exit(0)
	for row in X:
		p = Point(row)
		D.append(p)

	#label all that points
	for  i, point in enumerate(D):
		point.label = i

	eps = 1000
	MinPts = 10 
	dbs = DBSCAN().dbscan(D, eps, MinPts)

