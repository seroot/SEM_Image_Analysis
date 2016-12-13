# SEM_Image_Analysis
Python based image analysis code for nanoisland characterization


This code consists of 3 parts:

	1) Main.py
		Script defining how SEM files are read in and operated upon.
		run as:
		> python Main.py
		From a directory containing SEM images formatted as such:
		cu_au_110_300_100_1.tif
			1) Substrate (cu)
			2) Evaporant (au)
			3) Nominal thickness (110 (A))
			4) Deposition Temperature (300K)
			5) Magnification (100X)
			6) Index (1)

	2) SEM_IM.py
		- Module defining a SEM_IM class that contains all the 
		information pertaining to the image analysis of a single
		SEM image
		- Makes extensive use of Mahotas.py, Numpy.py, and Pyplot.py


	3) Plot.py
		- Plotting module for operating on a list of SEM_IM objects that
		have been serialized using Pickle.py
		- Plot aesthetics are done with Seaborn.py
