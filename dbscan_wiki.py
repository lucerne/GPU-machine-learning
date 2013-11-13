import sys
import csv
import numpy as np

class DBSCAN(object):
	"""docstring for DBSCAN"""
	def __init__(self, ):
		super(DBSCAN, self).__init__()

	def dbscan(self, D, eps, MinPts):
		C = 0
		for P in D and not P.visited:
			P.visited = True
			NeighborPts = self.regionQuery(P, eps)
			if sizeof(NeighborPts) < MinPts:
				P.NOISE = True
			else:
				C = []
				self.expandCluster(P, NeighborPts, C, eps, MinPts)
	       
	def expandCluster(self, P, NeighborPts, C, eps, MinPts):
		C.append(P)
		for P_dash in NeighborPts:
			if not P_dash.visited:
				P_dash.visited = True
				NeighborPts_dash = self.regionQuery(P_dash, eps)
				if sizeof(NeighborPts_dash) >= MinPts:
					NeighborPts = NeighborPts.extend(NeighborPts_dash)
				if not P_dash.isClusterMember:
					C.add(P_dash)
					P_dash.isClusterMember = True

	def in_range(self, point, P_dash):
		"""docstring for in_range"""
		return (math.sqrt((point-P_dash)**2) <= eps)
		
		  
	def regionQuery(self, P, eps):
		"""docstring for passre"""
		return [ point for point in D if in_range(point, P)]

			
if __name__ == '__main__':
	assert len(sys.argv) > 1, " Input file must be provided."
	input_file_name = sys.argv[1]
	D = np.float32(np.genfromtxt(input_file_name, delimiter=','))
	print D
	eps = 0.0003
	MinPts = 10
	dbs = DBSCAN().dbscan(D, eps, MinPts)
