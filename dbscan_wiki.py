import sys
import math
import csv
import numpy as np

MAX_NUM_OF_CLUSTERS = 10 
class Point(np.ndarray):

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
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
			if not P.visited:
				P.visited = True
				NeighborPts = self.regionQuery(P, eps)
				if len(NeighborPts) < MinPts:
					P.NOISE = True
				else:
					# pick new cluster
					count += 1
					c = C[count]
					self.expandCluster(P, NeighborPts, c, eps, MinPts)
	       
	def expandCluster(self, P, NeighborPts, c, eps, MinPts):
		c.append(P)
		NeighborPtsCopy = NeighborPts.copy()
		for P_dash in NeighborPts:
			if not P_dash.visited:
				P_dash.visited = True
				NeighborPts_dash = self.regionQuery(P_dash, eps)
				if len(NeighborPts_dash) >= MinPts:
					NeighborPts = NeighborPts.extend(NeighborPts_dash)
				if not P_dash.isClusterMember:
					c.append(P_dash)
					P_dash.isClusterMember = True

	def in_range(self, point, P_dash):
		"""docstring for in_range"""
		return np.sqrt(np.sum((point-P_dash)**2)) <= eps
		
		  
	def regionQuery(self, P, eps):
		"""docstring for regionQuery"""
		#return [point for point in D if self.in_range(point, P)]
		result =  [point for point in D if self.in_range(point, P)]
		if result is None:
			result = []
		print "***********************"
		print P
		print result
		print
		print "***********************"
		return result

			
if __name__ == '__main__':
	assert len(sys.argv) > 1, " Input file must be provided."
	input_file_name = sys.argv[1]
	X = np.genfromtxt(input_file_name, delimiter=',')
	D = Point(X.shape, dtype=np.float32)
	D[:] = X[:]
	eps = 0.3
	MinPts =1 
	dbs = DBSCAN().dbscan(D, eps, MinPts)

