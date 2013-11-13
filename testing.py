import numpy as np

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

if __name__ == '__main__':
	p = Point((2,), dtype='float32')
	print p.visited
	print p.isClusterMember
	a = np.array([1,2])
	p[:] = a
	print p

