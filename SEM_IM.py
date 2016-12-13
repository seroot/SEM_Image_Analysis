#!usr/bin/python
# Import necessary modules

import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import glob
import pylab
from pylab import rcParams
import seaborn as sns
from pylab import rcParams
import os
rcParams['figure.figsize'] = 12, 10
sns.set_style('ticks', {'xtick.direction': u'in', 'ytick.direction': u'in'})

scale_dict = { 80:2.4, 100:1.92, 125:1.42, 200:0.96, 250:0.77, 400:.48} # nm /  pixel^(1/2)


class SEM_IM(object):
    """
        Class defining an SEM image to be analyzed
    """
    #Constructor
    def __init__(self, filename):
        self.name = filename.strip('.tif').strip('.png')
        self.image = mh.imread(filename) # SEM image
        self.image_orig = mh.imread(filename)
        info = self.name.split('_')
        print info
    
        try:
            self.substrate = info[0]
            self.evaporant = info[1]
            
            try:
                self.thickness = int(info[2]) # Angstrom
                self.temperature = int(info[3])
                self.mag = int(info[4])
            except:
                try:
                    self.thickness = int(info[2].strip('A'))
                    self.temperature = int(info[3].strip('K'))
                    self.mag = int(info[4].strip('x'))
                except:
                    self.thickness = int(info[2].strip('nm'))
                    self.temperature = int(info[3].strip('K'))
                    self.mag = int(info[4].strip('x'))
            self.index = int(info[5])
        except:
            self.substrate = info[0]
            self.evaporant = info[1]
            self.thickness = int(info[2]) # Angstrom
            self.temperature = 300
            self.mag = int(info[4])
            self.index = int(info[5])

        self.scale = scale_dict[self.mag]

        self.thresh = 0.0 # numerical threshold to distingush islands from graphene

        # Print relevant information
        print "----------------------------------------\n"
        print "Reading in", filename
        print "Shape of image is: ", self.image.shape
        columns = self.image.shape[0]
        print "Data type is: ", self.image.dtype
        print "Maximum Value is: ", self.image.max()
        print "Minimum Value is: ", self.image.min()
        
        # TO DO: Determine SEM Parameters from filename
        # cu_au_5_300_100_1.tif

        return

    def crop_bottom(self):
        # Find the bottom of the SEM images in order to crop it
        i = 0
        try:
            for Line in self.image:
                if Line.all() == 0:
                    print "Found it"
                    Cutoff = i
                    print Cutoff
                    break
                else:
                    i += 1
        
            Cutoff = 484
            print "Cutoff is: ", Cutoff
            self.image = self.image[0:Cutoff, :] # Crop the picture
        except:
            print "No file"
                    

        return



    def run_analysis(self):
        # Define threshold
        self.thresh = mh.thresholding.otsu(self.image)
        self.thresh = mh.thresholding.rc(self.image) - 7
        print self.thresh
        self.labeled, self.nr_objects = mh.label(self.image > self.thresh)# Distinguish different Islands
        self.labeled_orig, self.nr_objects_orig = mh.label(self.image > self.thresh)# Distinguish different Islands
        self.sizes_orig = mh.labeled.labeled_size(self.labeled_orig)
        self.sizes = mh.labeled.labeled_size(self.labeled) # Find size distribution of islands
        self.total_area = self.labeled.shape[0]* self.labeled.shape[1] # Find total number of pixels
        self.coverage = 1 - (float(self.sizes[0]) / float(self.total_area)) #
        self.labeled = mh.labeled.remove_bordering(self.labeled) # Remove islands at the boarder
        too_big = np.where(self.sizes < 50) # Arbitrarily defined cutoff
        self.labeled  = mh.labeled.remove_regions(self.labeled, too_big)
        #self.labeled = mh.labeled.filter_labeled(self.labeled, remove_bordering= True, min_size = 2)
        self.labeled, self.n_left = mh.labeled.relabel(self.labeled, inplace = True) # Relabel islands
        self.sizes = mh.labeled.labeled_size(self.labeled)  # Find size distribution of islands
        self.borders = mh.labeled.borders(self.labeled) # Find the borders of the islands
        self.num_islands = len(self.sizes) - 1 # Find number of islands
        print self.labeled.shape[0]* self.labeled.shape[1]
        print self.sizes
        print self.num_islands # Print number of islands
        #print self.labeled.shape # print shape of image
        
        self.perimeters = np.zeros(self.num_islands , dtype= int) # Intitialzie empty array
        self.borders_int = np.zeros(self.labeled.shape, dtype=int) # Intialize empty array
        
        for i in range(self.labeled.shape[0]):
            for j in range(self.labeled.shape[1]):
                if self.borders[i,j].all():
                    self.perimeters[self.labeled[i,j]-1] += 1 # Creates an array of perimeter values for each island
                    self.borders_int[i,j] += 1 # Creates a matrix of 0 and 1 that sigifies the borders
    
        self.labeled_p, self.nr_objects = mh.label(self.borders >= 1) # label perimeters
        self.sizes_p = mh.labeled.labeled_size(self.labeled_p)
        # 2) Calculate the length of the perimeter
        self.size_dist = []
        self.perimeter_dist = []

        for element in self.sizes:
            if element < self.sizes[0] and element > 20:
                self.size_dist.append(element)
    
        for element in self.sizes_p:
            if element < self.sizes_p[0] and element > 20:
                self.perimeter_dist.append(element)
        
        print len(self.size_dist), len(self.perimeter_dist), len(self.perimeters)
        print self.perimeters
        print 'Background Size', self.sizes[0]

        self.fractal_dist = [] # Define a list of fractal coefficents (perimeter / area)
        for i in range(len(self.size_dist)):
            self.fractal_dist.append(self.perimeters[i]/self.size_dist[i])

        # Convert everything into a numpy array
        self.size_dist = np.asarray(self.size_dist, dtype = float)
        self.perimeter_dist = np.asarray(self.perimeter_dist, dtype = float)
        self.fractal_dist = np.asarray(self.fractal_dist, dtype = float)
        
        
        # Color by size
        self.sorted_indices = [ i[0] for i in sorted(enumerate(self.sizes_orig), key=lambda x:x[1])]
        print self.sorted_indices
        self.labeled_by_size= np.zeros(self.labeled_orig.shape, dtype=int)
        self.labeled_by_area = np.zeros(self.labeled_orig.shape, dtype=int)
        
        for i in range(self.labeled_orig.shape[0]):
           for j in range(self.labeled_orig.shape[1]):
               if self.labeled_orig[i,j].any() != 0:
                   #self.labeled_by_size[i,j] =  self.sorted_indices.index(self.labeled_orig[i,j]) + 1
                   self.labeled_by_area[i,j] = self.sizes_orig[self.labeled_orig[i,j]]
        """
        pylab.hot()
        plt.xticks([])
        plt.yticks([])
        pylab.imshow(self.labeled_by_area)
        plt.colorbar()
        plt.title(self.thresh)
        plt.show()
        
        """
                                                     
                                                    

                                                     
    
                                                     
        return

    def compute_rg(self):
        self.com = np.zeros([len(self.size_dist), 2], dtype = float )
        self.rg = np.zeros([len(self.size_dist), 2], dtype = float )
        self.rg_m = np.zeros(len(self.size_dist), dtype = float )
        # Compute the center of mass
        for i in range(self.labeled.shape[0]):
            for j in range(self.labeled.shape[1]):
                if self.labeled[i,j] != 0:
                    self.com[self.labeled[i,j] - 1 , 0] += i / self.size_dist[self.labeled[i,j] - 1]
                    self.com[self.labeled[i,j] - 1, 1] += j / self.size_dist[self.labeled[i,j] - 1]
        # Compute the radius of gyration
        for i in range(self.labeled.shape[0]):
           for j in range(self.labeled.shape[1]):
                if self.labeled[i,j] != 0:
                    self.rg[self.labeled[i,j] - 1, 0] += float(i - self.com[self.labeled[i,j] - 1, 0])**2 / self.size_dist[self.labeled[i,j] - 1]
                    self.rg[self.labeled[i,j] - 1, 1] += float(j - self.com[self.labeled[i,j] - 1, 1])**2 / self.size_dist[self.labeled[i,j] - 1]
                        
        plt.subplot(311)
        for i in range(len(self.rg)):
            self.rg_m[i] = np.sqrt(self.rg[i,0] + self.rg[i,1])
            #plt.plot(self.com[i,1], self.com[i,0], 'o', color='k', markersize = self.rg_m[i]/2.  )
                    
        print self.rg_m
        
 

        # Plot distinguished image
        """
        pylab.plasma()
        plt.xticks([])
        plt.yticks([])
        plt.title('%d seperate islands' %self.num_islands)
        pylab.imshow(self.labeled)

        plt.subplot(312)
        plt.hist(self.rg_m , normed = True, align = 'mid')
        plt.xlabel('Radius of Gyration')
        plt.ylabel('Probability')

        plt.subplot(313)
        plt.plot(self.size_dist, self.rg_m, 'o')
        plt.xlabel('Area [nm$^2$]')
        plt.ylabel('Rg [nm]')
        plt.show()
        
        """
        return

    def compute_rdf(self):
        RDF = []
        N = len(self.size_dist)
        dr = 10
        max = 500
        maxbin = int(max/dr)-1
        hist = np.zeros(maxbin, dtype= int)
        r = np.arange(dr, max, dr)
        self.rdf = np.zeros(maxbin, dtype=float)
        # Compute a histogram of distances between centers of masses
        for i in range(N):
            for j in range(N):
                if i != j:
                    vec = self.com[i] - self.com[j]
                    distance = np.sqrt(vec[0]**2 + vec[1]**2)
                    bin = int(distance/dr)
                    if bin <  maxbin:
                        hist[bin] += 1

        # TO DO: Normalize the RDF correctly
        for i in range(len(hist)):
            self.rdf[i] = float(hist[i])/float(((i+1)**2 - i**2)*dr**2)
        
        #plt.plot(self.rdf , '-', linewidth=3)
        #plt.xlabel('Distance')
        #plt.ylabel('Probability')
        #plt.show()
        return




    def plot_results(self):
        # Plot the cropped picture
        plt.subplot(421)
        plt.xticks([])
        plt.yticks([])
        plt.title(self.name)
        pylab.imshow(self.image_orig)
        pylab.gray()
        
        # Plot thresholded image
        pylab.gray()
        plt.subplot(422)
        plt.xticks([])
        plt.yticks([])
        plt.title('coverage = %.2f' % self.coverage)
        pylab.imshow(self.image > self.thresh)

        # Plot distinguished image
        pylab.hot()
        plt.subplot(423)
        plt.xticks([])
        plt.yticks([])
        plt.title('%d seperate islands' %self.num_islands)
        pylab.imshow(self.labeled_by_area)
        plt.colorbar()
        
        
        # Plot distinguished image
        plt.subplot(424)
        plt.xticks([])
        plt.yticks([])
        plt.title('%d seperate islands' %self.num_islands)
        pylab.imshow(self.labeled_p)
        
        # Plot histogram

        plt.subplot(425)
        plt.hist(self.size_dist*(1.92**2), normed = True, align = 'mid')
        plt.xlabel('Projected Area (nm$^2$)')
        plt.ylabel('Probability')
        
        # Plot histogram
        plt.subplot(426)
        plt.hist(self.perimeters[1:-1]*1.92 , normed = True, align = 'mid')
        plt.xlabel('Perimeter (nm)')
        plt.ylabel('Probability')

        
        # Plot histogram
        plt.subplot(427)
        plt.hist(self.perimeter_dist , normed = True, align = 'mid')
        plt.xlabel('Fractal Coeffecient')
        plt.ylabel('Probability')

       
        plt.subplot(428)
        plt.plot(self.perimeters[1:-1]*1.92, self.size_dist[1:-1]*1.92**2, 'o')
        plt.xlabel('Perimeter [nm]')
        plt.ylabel('Area [nm$^2$]')
        plt.savefig(self.name +".png")
        plt.show()
        
        """
        self.image_bool = input('Use this image? (True or False)')
        self.image_bool = bool(self.image_bool)
        if not self.image_bool:
            print 'image removed'
            os.system('rm %s.tif' % self.name)
        """
        
        return



