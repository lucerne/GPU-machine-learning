"""
CS205 final project: serial code
Name: Sytze Harkema 
Date: November 24, 2013

This scrip will import data, transform the data, and run the DBSCAN clustering
algorithm to cluster the data. 

"""

#______________________________________________________________________________
# imports:

#import matplotlib.pyplot as plt
import pylab as plt
from matplotlib import cm
import numpy as np
import csv
import sys
import time


#______________________________________________________________________________
# Definitions

# import data
def impData(filename):
    """
    impData opens a data csv file, determines size, and reads and processes
    the data into an np.array with float32's
    """
    # determine size data
    df = open(filename, 'rb') 
    # create buffered reader
    reader = csv.reader(df)
    try:
        for row in reader:
            # do nothing but parse to end of file
            0           

        # determine size
        countLine = reader.line_num     # length
        rowlength = len(row)            # width

        # create array for data
        data = np.zeros((countLine, rowlength), dtype=np.float32)

        # reset reader to start
        df.seek(0)
        # index
        i = 0

        for row in reader:
            data[i, :] = np.float32(row[:])
            i += 1
    except csv.Error as e:
        sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

    return data


# log transform data
def lgTransform(data):
    """
    Fluorecence data needs to be log transformed. 
    Log transform columns > 2
    """
    # create array
    nwData = np.empty_like(data)
    
    # copy first two colloms
    nwData[0,:] = data[0,:]
    nwData[1,:] = data[1,:]
    
    # for all other colloms
    for i in xrange(2,len(data[0,:])):
        nwData[:,i] = np.log10(data[:,i])
        
    return nwData



def serDBScan(data, eps, minPts):
    """
    Simple DBScan implementation O(N^2)
    """
    # number/name of clusters
    C = [0]
    arLen = data.shape[0]

    # create administration arrays
    visited = np.array([False]*arLen, dtype=bool)*False
    category = np.zeros((arLen,), dtype=np.int32)
    cluster = np.zeros((arLen,), dtype=np.int32)
    noNeigh = np.ones((arLen,), dtype=np.int32)

    # for each data point, the data point either is visited and belongs to an existing cluster,
    # or unvisited and belongs to a new cluster not explored yet, or belongs to noise
    for index, point in enumerate(data): ## loop differently over data
        # for unvisited points
        if visited[index] == True:
            continue
        # set visited
        visited[index] = True

        # perform neighborhood query 
        neighbors = regionQuery(point, data, eps)
        noNeigh[index] = len(neighbors)
        # check whether it is a cluster point
        if noNeigh[index]  < minPts:
            category[index] = NOISE
        else:
            C[0] += 1
            expandCluster(data, index, C, neighbors, visited,
                          cluster, category, noNeigh, eps, minPts)

    return category, cluster, noNeigh

      
def expandCluster(data, indexPoint, C, neighbors, visited,
                          cluster, category, noNeigh, eps, minPts):

    category[indexPoint] = CLUSTER
    cluster[indexPoint] = C[0]
    seedList = []
    seedList[:] = neighbors[:]
    seedInd = 0

    # while points in seed list expand total cluster
    while(len(seedList) > seedInd):
        pointNb = seedList[seedInd]

        if visited[pointNb] == False:
            # set visited
            visited[pointNb] = True
            # run region Query
            neighbors = regionQuery(data[pointNb], data, eps)
            # update no of neigbours
            noNeigh[pointNb] = len(neighbors)

            # evaluate category of seedpoint
            if noNeigh[pointNb] >= minPts:
                category[pointNb]= CLUSTER    
                # append non-evaluated neighbors
                for ind in neighbors:
                    if (visited[ind] == False) and (ind not in seedList):
                        seedList.extend([ind])
            else:
                category[pointNb] = BORDER

            # add to current cluster
            cluster[pointNb] = C[0]
            
        # go to next seed point
        seedInd += 1

def in_range(data_1, data_2, eps):
    """returns boolean wheter 2 points are within distance eps"""
    return np.sqrt(np.sum((data_1-data_2)**2)) <= eps
    
      
def regionQuery(point_A, data , eps):
    """creates a list of idices of neighbors"""
    return [ind for ind, point_B in enumerate(data) if in_range(point_A, point_B, eps)]


# display
def pltData(dataX, dataY, category):
    """
    pltData takes two arrays of data and plots these on two axis. 
    """
    # set limits
    x_min   = 80000  #np.amin(dataX)
    x_max   = 90000
    y_min   = 10000
    y_max   = 25000
    npts    = len(dataX)

    # plot data points.
    plt.figure()

    plt.scatter(dataX, dataY,c=category, cmap=cm.coolwarm)
    plt.title('scatter (%d points)' % npts)
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    #
    plt.show()




#______________________________________________________________________________
# GLOBALS

NOISE = -1
CLUSTER = 1
BORDER = 0


#______________________________________________________________________________
# Main

if __name__ == '__main__':
    # import data
    filename1 = "CG56_LCMV_total.csv"
    dataInp = impData(filename1)
    
    print "dataInp.shape", dataInp.shape

    # transform data
    nwData = lgTransform(dataInp)
    datL = nwData.shape[0]
    
    print "nwData.shape", nwData.shape    

    # reduce data size: why? 
    # dataFr = datL/1000

    # apply DBScan on the first two columns
    dataFr = datL
    dataCl = np.zeros((dataFr, 2), dtype=np.float32)
    dataCl[:,0] = nwData[:dataFr,0]
    dataCl[:,1] = nwData[:dataFr,1]

    print "dataCl.shape", dataCl.shape


    # run DBscan
    category, cluster, noNeigh = serDBScan(dataCl, 2000.0, 3)

    # make plot
    pltData(dataCl[:,0], dataCl[:,1], cluster)

##    # loop over various settings
##    eps = 1000.0
##    minClusterSize = 10
##    maxClNo = 0
##    while ( maxClNo < 5):
##        eps *= 2
##        # perform clustering
##        start = time.time()
##        N =perfDBScan(dataCl, eps, minClusterSize)[1]
##        end = time.time()
##        print "Step eps = %f took %f s" %(eps, end-start)
##        if N > maxClNo:
##            maxClNo = N
##            maxEps = eps
##        if eps > 4000:
##            break
##
##    # Select one to perform plot
##    db, N = perfDBScan(dataCl, 1000, 10)
##  
##    # Display clustering
##    pltDBRes(db, dataCl, N)
##    #pltData((nwData[:,0]),(nwData[:,1]))
##    #pltDataMulti(nwData)  
    print "The End"

#______________________________________________________________________________
# End
