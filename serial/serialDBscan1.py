"""
CS205 final project: serial code
Name: Sytze Harkema && Vinod Halaharvi
Date: November 10, 2013

This scrip will import data, transform the data, and run the DBSCAN clustering
algorithm to cluster the data. The target is to get a solid understanding
of the parameters which would make reasonable clustering, find a method
to display the data in a correct way, and to set a benchmark for the parallel
code. 



"""

#______________________________________________________________________________
# imports:
#import matplotlib.pyplot as plt
import pylab as plt
import numpy as np
import sklearn.cluster as cl
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


# perform dbscan
def perfDBScan(data, eps, minClusterSize):
    """
        Excecutes the DBSCAN algorithm
    """
    
    # Compute DBSCAN
    db = cl.DBSCAN(eps=eps, min_samples=minClusterSize).fit(data)

    core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Performed DBSCAN with minClustersize %d and distance %f" %(minClusterSize, eps))
    print('Estimated number of clusters: %d' % n_clusters_)
    return db, n_clusters_

def pltDBRes(db, data, n_clusters_):
    """
        The plot function is strongly inspired by:
        http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    """
    # set labels
    labels = db.labels_
    core_samples = db.core_sample_indices_
    # mark unique labels
    unique_labels = set(labels)
    # use colormap
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        
        class_members = [index[0] for index in np.argwhere(labels == k)]

        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        
        for index in class_members:
            x = data[index]
            if index in core_samples and k != -1:
                markersize = 14
            else:
                markersize = 6
            plt.plot(x[0], x[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=markersize)
            plt.xlim(50000, 100000)
            plt.ylim(10000, 40000)

    plt.title('Using skilearn package\nEstimated number of clusters: %d' % n_clusters_)
    plt.show()



# display
def pltData(dataX, dataY):
    """
    pltData takes two arrays of data and plots these on two axis. 
    """
    # set limits
    x_min   = 80000  #np.amin(dataX)
    x_max   = 90000
    y_min   = 80000
    y_max   = 90000
    npts    = len(dataX)

    # plot data points.
    plt.figure()
    plt.scatter(dataX, dataY)
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    plt.title('scatter (%d points)' % npts)
    plt.show()

def pltDataMulti(data):
    """
    pltData takes two arrays of data and plots these on two axis. 
    """

    npts    = data.shape[0]

    # plot data points.
    plt.figure()
    plt.title('scatter (%d points)' % npts)
    
    plt.subplot(241)
    dataX = data[:,0]
    dataY = data[:,1]
    # set limits
##    x_min   = np.amin(dataX)
##    x_max   = np.amax(dataX)
##    y_min   = np.amin(dataY)
##    y_max   = np.amax(dataY)
    plt.scatter(dataX, dataY)
##    plt.xlim(x_min, x_max)
##    plt.ylim(y_min, y_max)

    plt.subplot(242)
    dataX = data[:,2]
    dataY = data[:,3]
##    # set limits
##    x_min   = np.amin(dataX)
##    x_max   = np.amax(dataX)
##    y_min   = np.amin(dataY)
##    y_max   = np.amax(dataY)
    plt.scatter(dataX, dataY)
##    plt.xlim(x_min, x_max)
##    plt.ylim(y_min, y_max)
##
    plt.subplot(243)
    dataX = data[:,4]
    dataY = data[:,5]
##    # set limits
##    x_min   = np.amin(dataX)
##    x_max   = np.amax(dataX)
##    y_min   = np.amin(dataY)
##    y_max   = np.amax(dataY)
    plt.scatter(dataX, dataY)
##    plt.xlim(x_min, x_max)
##    plt.ylim(y_min, y_max)
##
    plt.subplot(244)
    dataX = data[:,6]
    dataY = data[:,7]
##    # set limits
##    x_min   = np.amin(dataX)
##    x_max   = np.amax(dataX)
##    y_min   = np.amin(dataY)
##    y_max   = np.amax(dataY)
    plt.scatter(dataX, dataY)
##    plt.xlim(x_min, x_max)
##    plt.ylim(y_min, y_max)

    plt.subplot(245)
    dataX = data[:,3]
    dataY = data[:,4]
    plt.scatter(dataX, dataY)

    plt.subplot(246)
    dataX = data[:,5]
    dataY = data[:,4]
    plt.scatter(dataX, dataY)

    plt.subplot(247)
    dataX = data[:,5]
    dataY = data[:,4]
    plt.scatter(dataX, dataY)

    plt.subplot(248)
    dataX = data[:,6]
    dataY = data[:,4]
    plt.scatter(dataX, dataY)

    plt.show()


#______________________________________________________________________________
# Main

if __name__ == '__main__':
    # import data
    filename1 = "CG56_LCMV_total.csv"
    #filename1 = "total.csv"
    filename1 = "head.csv"
    data = impData(filename1)

    # transform data
    #nwData = lgTransform(data)
    nwData = data 
    datL = data.shape[0]

    # reduce data size
    #dataFr = datL/16
    dataFr = datL
    print dataFr
    dataCl = np.zeros((dataFr, 2), dtype=np.float32)
    dataCl[:,0] = nwData[:dataFr,0]
    dataCl[:,1] = nwData[:dataFr,1]

    # loop over various settings
    eps = 512.0
    minClusterSize = 100
    maxClNo = 0
    while ( maxClNo < 5):
        eps *= 2
        # perform clustering
        start = time.time()
        N =perfDBScan(dataCl, eps, minClusterSize)[1]
        end = time.time()
        print "Step eps = %f took %f s" %(eps, end-start)
        if N > maxClNo:
            maxClNo = N
            maxEps = eps
        if eps > 4000:
            break

    # Select one to perform plot
    #db, N = perfDBScan(dataCl, 2048, 100)
    db, N = perfDBScan(dataCl, 1000, 10)
    # Display clustering
    pltDBRes(db, dataCl, N)
    #pltData((nwData[:,0]),(nwData[:,1]))
    #pltDataMulti(nwData)  
    print "The End"

#______________________________________________________________________________
# End
