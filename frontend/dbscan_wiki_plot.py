import sys
import math
import csv
import numpy as np
import copy 

MAX_NUM_OF_CLUSTERS = 10 
debug = True

eps = 0.3
MinPts =1


class Point(np.ndarray):

    def __new__(subtype, shape, dtype=np.float64, buffer=None, offset=0,
          strides=None, order=None, info=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, 
			strides, order)
	obj.visited = False
	obj.isClusterMember = False
        return obj

    def __array_finalize__(self,obj):
        self.visited = getattr(obj, 'visited', None)
        self.isClusterMember = getattr(obj, 'isClusterMember', None)


class DBSCAN(object):
	"""docstring for DBSCAN"""
	def __init__(self, ):
		super(DBSCAN, self).__init__()

	
	def dbscan(self, D, eps, MinPts):
		count = 0 
		C = [[] for _ in range(MAX_NUM_OF_CLUSTERS)]
		print C
		for P in D:
			assert self.in_range(P, P) == True
			if not P.visited:
				P.visited = True
				NeighborPts = self.regionQuery(P, eps, D)
				assert NeighborPts != None

				if len(NeighborPts) < MinPts:
					P.NOISE = True
				else:
					# pick new cluster
					count += 1
					c = C[count]
					self.expandCluster(P, NeighborPts, c, eps, MinPts, D)
	       
	def expandCluster(self, P, NeighborPts, c, eps, MinPts, D):
		c.append(P)
		assert NeighborPts != None
		for P_dash in copy.deepcopy(NeighborPts):
			if not P_dash.visited:
				P_dash.visited = True
				NeighborPts_dash = self.regionQuery(P_dash, eps, D)
				if len(NeighborPts_dash) >= MinPts:
					NeighborPts.extend(NeighborPts_dash)
				if not P_dash.isClusterMember:
					c.append(P_dash)
					P_dash.isClusterMember = True

	def in_range(self, point, P_dash):
		"""docstring for in_range"""
		return np.sqrt(np.sum((point-P_dash)**2)) <= eps
		
		  
	def regionQuery(self, P, eps, D):
		"""docstring for regionQuery"""
		result =  [point for point in D if self.in_range(point, P)]
		if debug:
			print "***********************"
			print P
			print "---"
			for item in result:
				print item
			print
		return result


