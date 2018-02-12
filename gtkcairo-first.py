import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.signal as sig
import scipy.stats as stats
import datetime
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib as mpl
mpl.use('GTKCairo')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import netCDF4 as nc4
from netCDF4 import date2num,num2date
import os
import sys
import ferr
from ferr import use
import mcmath
from mcmath import n2d,d2n
import mcp
import mcread
from subprocess import call,Popen,PIPE
from glob import glob
from numpy import *
