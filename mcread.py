"""A module containing functions for reading various files in various
formats; usually ASCII or binary data."""

import numpy as np
import scipy as sp
import scipy.signal as sig
import scipy.stats as stats
import datetime
from datetime import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt


def pdoread(datlist):
	"""A function for reading PDO text data files which can be copy-pasted from
	http://jisao.washington.edu/pdo/PDO.latest.
	Call with a list of contents of file (e.g., from file object f1,
	pdolist = f1.readlines()
	"""

	dt1 = np.empty((0,),dtype=object)
	pdo = np.empty((0,),dtype=np.float32)

	for i in xrange(len(datlist)):
		data = datlist[i].split()
		if len(data) < 1: continue
		if (data[0][0].isdigit()):
			yr = int(data[0])
			dapp = []
			for ll in xrange(len(data[1:])):
				dapp.append(dt(yr,ll+1,15))
			dapp = np.array(dapp,dtype=object)
			dt1 = np.append(dt1,dapp)
			data = np.float32(np.array(data[1:]))
			pdo = np.append(pdo,data)

	return [dt1,pdo]



def yrmthread(datlist):
	"""A function for reading TXT text data files which has years running down
	the left column, months across the top and data in an array cross-ref'ing them.
	Call with a list of contents of file (e.g., from file object f1,
	txtlist = f1.readlines()
	"""

	dt1 = np.empty((0,),dtype=object)
	txt = np.empty((0,),dtype=np.float32)

	for i in xrange(len(datlist)):
		data = datlist[i].split()
		if len(data) < 1: continue
		if (data[0][0].isdigit()):
			yr = int(data[0])
			dapp = []
			for ll in xrange(len(data[1:])):
				dapp.append(dt(yr,ll+1,15))
			dapp = np.array(dapp,dtype=object)
			dt1 = np.append(dt1,dapp)
			data = np.float32(np.array(data[1:]))
			txt = np.append(txt,data)

	return [dt1,txt]

def txtread(txt1,delim=None,dtype='|S16'):
	"""A function for reading TXT text data files into an array.  Generally reads data
	'txt1', which is already ouput from some file's .readlines() function call,
	and parses them into a 2-d array as in the file (but separated by the delimiting
	symbol)."""

	nc = len(txt1[0].strip().split(delim))
	nr = len(txt1)

	txt2 = np.empty((nr,nc),dtype=dtype)

	for i in xrange(len(txt1)):
		data = txt1[i].strip().split(delim)
		if len(data) < 1: continue
		data = np.array(data).reshape((1,-1))
		txt2[i,:] = data[:]

	return txt2

def mei_read(datlist):
	"""A function for reading MEI text data files which has years running down
	the left column, bimonths across the top and data in an array cross-ref'ing them.
	Call with a list of contents of file (e.g., from file object f1,
	txtlist = f1.readlines()
	"""

	dt1 = np.empty((0,),dtype=object)
	txt = np.empty((0,),dtype=np.float32)

	for i in xrange(len(datlist)):
		data = datlist[i].split()
		if len(data) < 1: continue
		if (data[0][0].isdigit()):
			yr = int(data[0])
			dapp = []
			for ll in xrange(len(data[1:])):
				dapp.append(dt(yr,ll+1,1))
			dapp = np.array(dapp,dtype=object)
			dt1 = np.append(dt1,dapp)
			data = np.float32(np.array(data[1:]))
			txt = np.append(txt,data)

	return [dt1,txt]

def iodread(datlist):
	"""Reads IOD data file, or any file with years in left column,
	months in second column, data in third column.
	Call with a list of contents of file (e.g., from file object f1,
	txtlist = f1.readlines()
	"""

	listsize = len(datlist)
	yr = np.empty((listsize,))
	mth = np.empty((listsize,))
	iod = np.empty((listsize,))
	day = np.ones((listsize,))
	day = day*15
	day = np.int16(day)

	for i in xrange(listsize):
		a = datlist[i].strip()
		b,c,d = a.split()
		yr[i] = int(b)
		mth[i] = int(c)
		iod[i] = float(d)
		dts[i] = dt(yr[i],mth[i],day[i])

	return iod,dts

def chur_ssh_read(filename):
	"""Reads Church / White 2006 SSH anomaly datafile.  Differs from other
	read functions in this module, in that only the file name is needed (to ensure
	the path is correct, it is not hard-coded here)."""

	f1 = open(filename)
	txtlist = f1.readlines()
	datlen = len(txtlist)

	yr = np.zeros(datlen,dtype=object)
	ssh = np.zeros(datlen,dtype=np.float32)
	err = np.zeros(datlen,dtype=np.float32)

	for i in xrange(datlen):
		yrtxt,sshtxt,errtxt = txtlist[i].strip().split()
		yr1 = np.float32(yrtxt)
		yr2 = dt(np.int16(yr1),1,1)
		delt = datetime.timedelta((yr1 - np.int16(yr1)) * 365)
		yr[i] = yr2 + delt
		ssh[i] = np.float32(sshtxt)
		err[i] = np.float32(errtxt)

	return [yr,ssh,err]


def spanread(flist,var1,k=None):
	"""reads a var from multiple files (different time intervals so far) and
	cats together the data into one array.

	Parameters:\tflist -- string with wildcards for files
	\t\tvar1 -- string name of data variable """

	from ferr import use
	from glob import glob
	blah = glob(flist)
	list1 = np.sort(blah)
	sizes = []


	if k is None:
		for f1 in xrange(list1.size):
			d1 = use(list1[f1],silent=1)
			sizes.append(d1.v[var1].shape[0])
			xshp = d1.v[var1][:].shape
			d1.f.close()
			del d1

		sizes = np.array(sizes,dtype=np.int16)
		xshp = np.array(xshp,dtype=np.int16)
		totsiz = sizes.sum()

		xshp[0] = totsiz
		vout = np.ma.masked_all(shape=xshp,dtype=np.float32)
		tout = np.zeros(totsiz,dtype=np.float64)
		start = 0

		for f1 in xrange(list1.size):
			d1 = use(list1[f1],silent=1)
			t1 = d1.getax(var1,'t')
			v1 = d1.v[var1][:]
			if isinstance(v1,np.ma.MaskedArray):
				v1_fill = d1.v[var1][:].get_fill_value()

			tout[start:start+sizes[f1]] = t1.copy()
			if isinstance(v1,np.ma.MaskedArray):
				vout[start:start+sizes[f1]] = v1.data.copy()
			else:
				vout[start:start+sizes[f1]] = v1.copy()

			if isinstance(v1,np.ma.MaskedArray):
				vout = np.ma.masked_values(vout,v1_fill)

			start = start + sizes[f1]
			del v1,t1
			d1.f.close()
			del d1

	else:
		for f1 in xrange(list1.size):
			d1 = use(list1[f1],silent=1)
			sizes.append(d1.v[var1].shape[0])
			xshp = d1.v[var1][:,k,:,:].shape
			d1.f.close()
			del d1

		sizes = np.array(sizes,dtype=np.int16)
		xshp = np.array(xshp,dtype=np.int16)
		totsiz = sizes.sum()

		xshp[0] = totsiz
		vout = np.ma.masked_all(shape=xshp,dtype=np.float32)
		tout = np.zeros(totsiz,dtype=np.float64)
		start = 0

		for f1 in xrange(list1.size):
			d1 = use(list1[f1],silent=1)
			t1 = d1.getax(var1,'t')
			v1 = d1.v[var1][:,k,:,:]
			if isinstance(v1,np.ma.MaskedArray):
				v1_fill = d1.v[var1][:,k,:,:].get_fill_value()

			tout[start:start+sizes[f1]] = t1.copy()
			if isinstance(v1,np.ma.MaskedArray):
				vout[start:start+sizes[f1]] = v1.data.copy()
			else:
				vout[start:start+sizes[f1]] = v1.copy()

			if isinstance(v1,np.ma.MaskedArray):
				vout = np.ma.masked_values(vout,v1_fill)

			start = start + sizes[f1]
			del v1,t1
			d1.f.close()
			del d1

	del sizes,xshp,totsiz,start

	return (tout,vout)

def psmsl(filename,dir='/scratch/local1/u241180/data/PSMSL/rlr_annual/data/'):

	fname = dir + filename + '.rlrdata'
	f1 = open(fname)
	f1txt = f1.readlines()
	blah = txtread(f1txt,';')

	yr = np.int16(blah[:,0])
	yr1 = np.ndarray(shape=yr.shape,dtype=object)
	for i in xrange(yr.size):
		yr1[i] = dt(yr[i],7,1)

	msl = np.float32(blah[:,1])
	msl = np.ma.masked_values(msl,-99999.)

	return [yr1,msl]


def m_psmsl(filename,dir='/scratch/local1/u241180/data/PSMSL/rlr_monthly/data/',qc=False):

	fname = dir + filename + '.rlrdata'
	f1 = open(fname)
	f1txt = f1.readlines()
	blah = txtread(f1txt,';')

	yrmth = np.float32(blah[:,0])
	mth1,yr1 = np.modf(yrmth)
	mth2 = np.int16(mth1*12) + 1
	dt1 = np.ndarray(shape=yr1.shape,dtype=object)
	for i in xrange(yr1.size):
		dt1[i] = dt(yr1[i],mth2[i],15)

	msl = np.float32(blah[:,1])
	msl = np.ma.masked_values(msl,-99999.)
	if qc == True:
		fblah = np.float32(blah[:,-1])
		indqc = np.where(fblah == 1)
		msl[indqc] = np.ma.masked

	return [dt1,msl]

def m_to_a_psmsl(filename,dir='/scratch/local1/u241180/data/PSMSL/rlr_monthly/data/',qc=True):

	import mcmath
	
	dt1,msl = m_psmsl(filename,dir,qc)
	start_dt = 0
	for i in xrange(dt1.size):
		if dt1[start_dt].month == 1:
			break
		else:
			start_dt += 1

	yrs = dt1[start_dt:].size / 12
	yrlen = yrs * 12
	dt1 = dt1[start_dt:yrlen+start_dt]
	msl = msl[start_dt:yrlen+start_dt]

	msldt = np.squeeze(mcmath.my_dtrnd(msl.reshape(-1,1,1),dt1))

	clim1 = mcmath.make_clim(msldt)
	climt = np.tile(clim1,yrs)

	i=0
	while i < msl.size-1:
		if msl[i] is np.ma.masked:
			i = i+1
			continue
		if msl.mask[i+1] != True:
			i = i+1
			continue
		else:
			extent=0
			while (msl.mask[i+1+extent]) == True:
				extent=extent+1
				if (i+1+extent) == msl.size:
					break
			endpt = i + 1 + extent
			if (endpt) == msl.size:
				break
			#print 'extent is %d\n' % extent
			if extent > 24:
				i = endpt
				continue
			clim2 = climt[i:endpt+1]
			a1 = msl[i] - clim2[0]
			b1 = (msl[endpt] - (clim2+a1)[-1]) / (endpt - i)
			c1 = np.arange(endpt - i + 1,dtype=np.float32)
			msl[i:endpt+1] = b1*c1 + clim2+a1
			i = endpt
			#print i

	dt1 = dt1[6::12]
	msl = mcmath.mth_ann_mean(msl)

	return [dt1,msl]



### May not need this function... but may yet.
##def mpi_spot(flist,var1,xind,yind):
##	"""Read an x,y spot of data from MPI Millenium data files.  Give filelist 'flist', variable 'var1', and
##	indices for x and y, 'xind' and 'yind'.  Returns depth-time array at that location."""

##	from ferr import use
##	from glob import glob
##	blah = glob(flist)
##	list1 = np.sort(blah)
##	sizes = []

##	for f1 in xrange(list1.size):
##		d1 = use(list1[f1],silent=1)
##		sizes.append(d1.d['time'].size)
##		xshp = d1.v[var1][:].shape
##		d1.f.close()
##		del d1

##	sizes = np.array(sizes,dtype=np.int16)
##	xshp = np.array(xshp,dtype=np.int16)
##	totsiz = sizes.sum()

##	xshp[0] = totsiz
##	vout = np.ma.masked_all(shape=xshp,dtype=np.float32)
##	tout = np.zeros(totsiz,dtype=np.float64)
##	start = 0

##	for f1 in xrange(list1.size):
##		d1 = use(list1[f1],silent=1)
##		t1 = d1.d['time'][:]
##		v1 = d1.v[var1][:]
##		v1_fill = d1.v[var1][:].get_fill_value()

##		tout[start:start+sizes[f1]] = t1.copy()
##		vout[start:start+sizes[f1]] = v1.data.copy()

##		vout = np.ma.masked_values(vout,v1_fill)

##		start = start + sizes[f1]
##		del v1,t1
##		d1.f.close()
##		del d1


	
##	return None
