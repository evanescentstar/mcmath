"""This module provides a variety of functions
that gather and process data from various NOAA et al. websites,
and also provides methods for comparing these observational data to
supplied model data.

The main functions that are used include:

ndbcget()
buoycoll()
wlget()
wlcoll()
wlqckcomp()
wlqckcomp2()
"""

# Imports
import pandas as pd
import xarray as xr
import numpy as np
import scipy as sp
from shapely import geometry
import requests
import cartopy.io.shapereader as shpreader
import re
import sys
import os
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

def poly_region(region_in, latlon=True, verbose=False):
    """Returns a shapely polygon defined either by the boundary of a grid from a netcdf grid file (so, a
    filename / filepath) or a tuple of 4 float numbers representing lower left lat-lon and
    upper right lat-lon, thus defining the polygon of a lat-lon box. If parameter 'latlon' is False, then
    the coordinates are treated as regular x,y in e.g. a Mercator projection (instead of y,x, lat then lon)."""

    if type(region_in) is str:
        gridfile = region_in
        boundbox = None
    elif (type(region_in) is tuple) and (len(region_in) == 4):
        boundbox = region_in
        gridfile = None
    else:
        raise Exception("'region_in' parameter must be either a string of a filename / filepath or "
                        "a tuple of 4 float numbers representing lower left lat-lon and upper right lat-lon "
                        "defining a region via a lat-lon box")
    if gridfile is not None:
        grd = xr.open_dataset(gridfile)
        grd = grd.set_coords(('lat_rho', 'lon_rho'))
        if 'h' in list(grd):
            grdH = grd.h.load()
            grdH = grdH.rename({"lon_rho": "lon", "lat_rho": "lat"})
        elif 'zeta' in list(grd):
            grdH = grd.zeta.load()
            grdH = grdH.rename({"lon_rho": "lon", "lat_rho": "lat"})

        glon = np.concatenate(
            (
                grdH.lon[0, :].values, grdH.lon[:, -1].values, np.flip(grdH.lon[-1, :].values),
                np.flip(grdH.lon[:, 0].values)))
        glat = np.concatenate(
            (
                grdH.lat[0, :].values, grdH.lat[:, -1].values, np.flip(grdH.lat[-1, :].values),
                np.flip(grdH.lat[:, 0].values)))
        llbounds = np.column_stack((glon, glat))
        p1 = geometry.Polygon(llbounds)

        return p1

    elif boundbox is not None:
        if latlon:
            y1, x1, y2, x2 = boundbox
            if verbose:
                print(f'y1: {y1}')
                print(f'x1: {x1}')
                print(f'y2: {y2}')
                print(f'x2: {x2}')
            xy = ((x1, y1), (x2, y2))
            axy = np.array(xy)
            xs = [axy[0, 0], axy[1, 0], axy[1, 0], axy[0, 0]]
            ys = [axy[0, 1], axy[0, 1], axy[1, 1], axy[1, 1]]
            p1 = geometry.Polygon(list(zip(xs, ys)))
        else:
            x1, y1, x2, y2 = boundbox
            if verbose:
                print(f'y1: {y1}')
                print(f'x1: {x1}')
                print(f'y2: {y2}')
                print(f'x2: {x2}')
            xy = ((x1, y1), (x2, y2))
            axy = np.array(xy)
            xs = [axy[0, 0], axy[1, 0], axy[1, 0], axy[0, 0]]
            ys = [axy[0, 1], axy[0, 1], axy[1, 1], axy[1, 1]]
            p1 = geometry.Polygon(list(zip(xs, ys)))

        return p1

    else:
        print('both gridfile and boundbox are not defined')
        raise 'no boundary specified'


def bfind(region_in, lats, lons):
    """bfind - Buoy find, although it'll find any set of lat/lons within a defined polygon.
    Input a region defined either by the boundary of a grid from a netcdf grid file (so, a
    filename / filepath) or a tuple of 4 float numbers representing lower left lat-lon and
    upper right lat-lon, thus defining a region via a lat-lon box, and equal sets of lats and lons.
    Returns the boolean index array that yield the indices of lat/lon values that are contained within the polygon.
    Usage: e.g., index_out = bfind('Mexico_Beach_grd2.nc', lats, lons)
            or   index_out = bfind((24.43, -74.32, 25.76, -72.87), lats, lons)

    """

    if lats.shape[0] != lons.shape[0]:
        raise Exception('lats and lons must be 1-d vectors and have equal length')

    if isinstance(region_in,geometry.Polygon):
        poly1 = region_in
    else:
        poly1 = poly_region(region_in)

    inside = np.zeros((lats.shape[0],), dtype=bool)

    for i in np.ndindex(lats.shape):
        pt1 = geometry.Point(lons[i], lats[i])
        if poly1.intersects(pt1):
            inside[i] = True

    return inside
# END of bfind()


def datestrnorm(dstr):
    """Requires at least a string of 'year-month-day' hyphen-delimited.  Fixes day or month which
    is a single digit to be zero-padded to 2 digits and a year which is just in 2-digit format
    to be in full 4 digit format.  This is needed to properly convert date strings into numpy datetime64
    objects so that they can be compared to xarrays / pandas datetime64 objects.

    e.g., datestrnorm('23-4-7') will return '2023-04-07'
    """
    from datetime import datetime as dt
    if len(dstr.split('-')[0]) == 2:
        outdstr = dt.strptime(dstr, '%y-%m-%d').strftime('%Y-%m-%d')
    elif len(dstr.split('-')[0]) == 4:
        outdstr = dt.strptime(dstr, '%Y-%m-%d').strftime('%Y-%m-%d')
    else:
        print("date string needs to be in format 'year-month-day'")
        return None
    return outdstr
## END of datestrnorm()

def ndbcread(ddset, btype_in, timefr, varset, verbose):
    ds1out = xr.Dataset({})
    timecoord = ddset.time[timefr].values
    if btype_in == 'swden':
        freqcoord = ddset.frequency.values
    for v1 in varset[btype_in]:
        vout = ddset.variables[v1][timefr].squeeze()
        if np.count_nonzero(~np.isnan(vout.values)) < 2:
            if verbose:
                print(f'var {v1} contains less than 2 values, skipping')
            continue
        elif btype_in == 'swden':
            ds1out[v1] = xr.DataArray(data=vout.values,
                                   dims=list(vout.dims),
                                   coords={'time': timecoord, 'frequency': freqcoord},
                                   attrs=vout.attrs)
        else:
            ds1out[v1] = xr.DataArray(data=vout.values,
                                   dims=list(vout.dims),
                                   coords={'time': timecoord},
                                   attrs=vout.attrs)
    return ds1out
## END of ndbcread()

def ndbcget(buoy, btype, time_frame, verbose=False):
    """Grabs the buoy data from the dods.ndbc.noaa.gov THREDDS webserver for buoy with id 'buoy'
    and type 'btype' (one of 'stdmet', 'swden', 'ocean', 'pwind', or 'cwind', and within the time frame
    in the form 'yyyy-mm-dd:yyyy-mm-dd' (in UTC time zone), such as '2022-03-05:2022-04-29', or through to the latest
    data, '2023-03-15:latest'.

    E.g., ndbcget('41112', 'stdmet', '2022-03-15:2022-04-29')
    """
    import datetime
    from datetime import datetime as dt
    start = time_frame.split(':')[0]
    end = time_frame.split(':')[1]
    print(f'looking up buoy {buoy} in {btype}, over the dates {start} to {end}')
    start = datestrnorm(start)
    if end in ['latest', 'present', 'today']:
        end = (dt.today() + datetime.timedelta(1)).strftime('%Y-%m-%d 23:59:59')
    else:
        end = datestrnorm(end)
    # INIT vars
    ds1 = None
    varset = {'stdmet': ['wind_dir', 'wind_spd', 'gust', 'wave_height', 'dominant_wpd', 'average_wpd', 'mean_wave_dir',
                         'air_pressure', 'air_temperature', 'sea_surface_temperature', 'dewpt_temperature',
                         'visibility', 'water_level'],
              'ocean': ['depth', 'water_temperature'],
              'swden': ['spectral_wave_density', 'mean_wave_dir', 'principal_wave_dir', 'wave_spectrum_r1',
                        'wave_spectrum_r2'],
              'pwind': ['gust_dir', 'gust_spd', 'gust_time'],
              'cwind': ['wind_dir', 'wind_spd']}
    suff = '/catalog.html'

    if btype not in varset.keys():
        if verbose:
            print(f"ERROR: buoy type '{btype}' not found")
            print(f"must be one of: {varset.keys()}")
        return None
    prefdata = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/' + btype + '/'
    prefhtml = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/' + btype + '/'
    dir1 = prefhtml + buoy + suff
    ddir1 = pd.read_html(dir1)[0]

    yr1 = start[:4]
    if end[:4] != yr1:
        if verbose:
            print('multi-year timeframes not yet supported')

    pack1 = ['9999',  yr1+'\.nc$', 'ncml$']

    for str1 in pack1:
        if len(ddir1.Dataset[ddir1.Dataset.str.contains(str1)]) != 0:
            if verbose:
                print(f'buoy {buoy} in {btype}')
            # ds1 = xr.Dataset({})
            dset = ddir1.Dataset[ddir1.Dataset.str.contains(str1)]
            dset1 = prefdata + buoy + '/' + dset.values[0]
            ddset1 = xr.open_dataset(dset1)
            if (ddset1.time[0] > np.datetime64(start)) | (ddset1.time[-1] < np.datetime64(start)):
                continue
            timefr = (ddset1.time <= np.datetime64(end)) & (ddset1.time >= np.datetime64(start))
            if timefr.values.sum() > 0:
                if verbose:
                    print(f'found {timefr.values.sum()} values in time frame')
                ds1 = ndbcread(ddset1, btype, timefr, varset=varset, verbose=verbose)
                break
            if str1 == yr1:
                if verbose:
                    print(f'buoy {buoy} has no data within this time frame', flush=True)
                sys.stdout.flush()
        elif str1 == yr1:
            if verbose:
                print(f'buoy {buoy} has no data within this time frame, or buoy not found', flush=True)
            sys.stdout.flush()
    if (ds1 is not None):
        if len(list(ds1.variables)) > 0:
            ds1.attrs = ddset1.attrs
            ds1['latitude'] = xr.DataArray(data=ddset1.latitude.values, dims=['latitude'], attrs=ddset1.latitude.attrs)
            ds1['longitude'] = xr.DataArray(data=ddset1.longitude.values, dims=['longitude'],
                                            attrs=ddset1.longitude.attrs)
        else:
            if verbose:
                print(f'no variables with data found for buoy {buoy} in {btype}')
            return None
    return ds1
## END of ndbcget()


def buoycoll(region_in, time_frame=None, outfile=None, variable=None, verbose=False):
    """buoycoll - 'buoy collect', a script which collects all NDBC buoys which lie within a specified region
    and contain data within a specified time range.

    Parameters:
          region_in  - input a region defined either by the boundary of a grid from a netcdf grid file (so, a
                       filename / filepath), or a tuple of 4 float numbers representing lower left lat-lon and
                       upper right lat-lon, thus defining a region via a lat-lon box
          time_frame - a time frame in the form 'yyyy-mm-dd:yyyy-mm-dd' (in UTC time zone);
                       NOTE: If 'region_in' is a netCDF dataset, and 'variable' is set to a named variable
                       belonging to the dataset, time_frame can be left undefined, and the time limits on the 
                       variable will be referenced from the dataset. Otherwise, this parameter is REQUIRED.
          outfile    - an optional argument that, when provided as a string, will output the data into a netcdf file
                       with the filename given by the string

    Returns the boolean index array that yield the indices of lat/lon values that are contained within the polygon.
    Usage: e.g., dataset = buoycoll('Mexico_Beach_grd2.nc', '2023-02-24:2023-02-28', 'buoys1.nc')
            or   dataset = buoycoll((24.43, -74.32, 25.76, -72.87), '2022-07-10:2022-10-18')"""

    # Check whether time_frame is given or can be defined
    if time_frame is None:
        if variable is None:
            print("The parameter 'time_frame' must be set, or if not, 'region_in' must be a netCDF file and 'variable' must be"
                "a variable that is defined in that file with a corresponding time coordinate.")
            return None
        dmodel = xr.open_dataset(region_in)
        t1 = dmodel[dmodel[variable].time]
        if isinstance(t1[0].values,np.datetime64):
            time_frame = str(t1[0].values).split('T')[0] + ":" + str(t1[-1].values).split('T')[0]
        else:
            print(f"can not find time data for variable {variable} in {region_in}")
            return None
    # gather the list of active stations, and their lons / lats
    dact = pd.read_xml('https://www.ndbc.noaa.gov/activestations.xml')
    lats = dact.lat.values
    lons = dact.lon.values

    # supply the region of interest and the lat-lon values of active stations to bfind
    # then extract the region-included stations
    outind = bfind(region_in, lats, lons)
    dbuoy = dact[outind]

    buoy1 = dbuoy.id.values
    buoylen = len(buoy1)
    if buoylen == 0:
        print(f'no buoys found in the region in given by {region_in}')
        sys.exit(1)



    print(buoy1)
    print(f'time frame: {time_frame}')

    # grab directory contents for the main buoy directories (the ones with actively reporting buoys for the most part)
    smname = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/stdmet/catalog.html'
    ocname = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/ocean/catalog.html'
    pwname = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/pwind/catalog.html'
    swname = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/swden/catalog.html'
    cwname = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/cwind/catalog.html'

    dsm = pd.read_html(smname)[0]
    doc = pd.read_html(ocname)[0]
    dpw = pd.read_html(pwname)[0]
    dsw = pd.read_html(swname)[0]
    dcw = pd.read_html(cwname)[0]

    dsm.replace('/', '', inplace=True, regex=True)
    doc.replace('/', '', inplace=True, regex=True)
    dpw.replace('/', '', inplace=True, regex=True)
    dsw.replace('/', '', inplace=True, regex=True)
    dcw.replace('/', '', inplace=True, regex=True)


    # initialize new collection of xr.DataArrays for all the buoys found
    ds1 = xr.Dataset({})

    # Loop through the buoys found, and gather the outputs within the time frame specified into the xarray Dataset ds1
    for buoy in buoy1:
        if buoy in dsm.Dataset.values:
            dout = ndbcget(buoy, 'stdmet', time_frame, verbose)
            if dout is None:
                pass
            elif len(dout.variables) == 0:
                pass
            else:
                for v1 in list(dout):
                    vout = dout[v1]
                    ds1[v1 + '_' + buoy] = xr.DataArray(data=vout.values,
                                                        dims=['time_' + buoy],
                                                        coords={'time_' + buoy: dout.time.values},
                                                        attrs=vout.attrs)
                loc = np.array([dout.latitude.values, dout.longitude.values]).reshape(2, )
                ds1['buoy_' + buoy] = xr.DataArray(data=loc,
                                                   dims=['location'],
                                                   attrs=dout.attrs)
                ds1['buoy_' + buoy].attrs['standard_name'] = 'lat,lon'

        if buoy in doc.Dataset.values:
            dout = ndbcget(buoy, 'ocean', time_frame, verbose)
            if dout is None:
                pass
            elif len(dout.variables) == 0:
                pass
            else:
                for v1 in list(dout):
                    vout = dout[v1]
                    ds1[v1 + '_' + buoy] = xr.DataArray(data=vout.values,
                                                        dims=['time_' + buoy + '_oc'],
                                                        coords={'time_' + buoy + '_oc': dout.time.values},
                                                        attrs=vout.attrs)
                if 'buoy_' + buoy not in ds1.variables:
                    loc = np.array([dout.latitude.values, dout.longitude.values]).reshape(2, )
                    ds1['buoy_' + buoy] = xr.DataArray(data=loc,
                                                       dims=['location'],
                                                       attrs=dout.attrs)
                    ds1['buoy_' + buoy].attrs['standard_name'] = 'lat,lon'

        if buoy in dsw.Dataset.values:
            dout = ndbcget(buoy, 'swden', time_frame, verbose)
            if dout is None:
                pass
            elif len(dout.variables) == 0:
                pass
            else:
                for v1 in list(dout):
                    vout = dout[v1]
                    ds1[v1 + '_' + buoy] = xr.DataArray(data=vout.values,
                                                        dims=['time_' + buoy + '_sw', 'frequency_' + buoy],
                                                        coords={'time_' + buoy + '_sw': dout.time.values,
                                                                'frequency_' + buoy: dout.frequency.values},
                                                        attrs=vout.attrs)
                if 'buoy_' + buoy not in ds1.variables:
                    loc = np.array([dout.latitude.values, dout.longitude.values]).reshape(2, )
                    ds1['buoy_' + buoy] = xr.DataArray(data=loc,
                                                       dims=['location'],
                                                       attrs=dout.attrs)
                    ds1['buoy_' + buoy].attrs['standard_name'] = 'lat,lon'

        if buoy in dpw.Dataset.values:
            dout = ndbcget(buoy, 'pwind', time_frame, verbose)
            if dout is None:
                pass
            elif len(dout.variables) == 0:
                pass
            else:
                for v1 in list(dout):
                    vout = dout[v1]
                    ds1[v1 + '_' + buoy] = xr.DataArray(data=vout.values,
                                                        dims=['time_' + buoy + '_pw'],
                                                        coords={'time_' + buoy + '_pw': dout.time.values},
                                                        attrs=vout.attrs)
                if 'buoy_' + buoy not in ds1.variables:
                    loc = np.array([dout.latitude.values, dout.longitude.values]).reshape(2, )
                    ds1['buoy_' + buoy] = xr.DataArray(data=loc,
                                                       dims=['location'],
                                                       attrs=dout.attrs)
                    ds1['buoy_' + buoy].attrs['standard_name'] = 'lat,lon'

        if buoy in dcw.Dataset.values:
            dout = ndbcget(buoy, 'cwind', time_frame, verbose)
            if dout is None:
                pass
            elif len(dout.variables) == 0:
                pass
            else:
                for v1 in list(dout):
                    vout = dout[v1]
                    ds1[v1 + '_' + buoy] = xr.DataArray(data=vout.values,
                                                        dims=['time_' + buoy + '_cw'],
                                                        coords={'time_' + buoy + '_cw': dout.time.values},
                                                        attrs=vout.attrs)
                if 'buoy_' + buoy not in ds1.variables:
                    loc = np.array([dout.latitude.values, dout.longitude.values]).reshape(2, )
                    ds1['buoy_' + buoy] = xr.DataArray(data=loc,
                                                       dims=['location'],
                                                       attrs=dout.attrs)
                    ds1['buoy_' + buoy].attrs['standard_name'] = 'lat,lon'

    if len(ds1) == 0:
        print(f'NO data were found for any buoys in the region and time frame given.')
    elif outfile is not None:
        ds1.to_netcdf(outfile)

    return ds1


def wlget(station, time_frame, datum='NAVD', units='metric', product='water_level', format='csv'):
    """
    product: one of 'predictions', 'water_level', 'one_minute_water_level', 'wind', and more
    datum: one of 'NAVD', 'MSL', 'MTL', and more
    format: one of 'csv', 'json', 'xml'
    units: one of 'metric' or 'english' (i.e., meters, or feet)
    """
    start = time_frame.split(':')[0].replace('-', '')
    end = time_frame.split(':')[1].replace('-', '')
    if type(station) is not str:
        stat1 = station
        station = '%d' % station
    else:
        stat1 = int(station)
    srchstr = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=' + product + '&application=NOPP_Team4' \
              '&begin_date=' + start + '&end_date=' + end + '&datum=' + datum + '&station=' + station + \
              '&time_zone=GMT&units=' + units + '&format=' + format
    try:
        duh = pd.read_csv(srchstr)
    except:
        print(f'no data returned for station {station}')
        return None
    if product == 'water_level':
        if duh[' Water Level'].count() == 0:
            print(f'station {station} contains zero data')
            return None
    elif product == 'predictions':
        if duh[' Prediction'].count() == 0:
            print(f'station {station} contains zero data')
            return None
    dact1 = pd.read_xml(
        'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml?type=waterlevels&units=english')
    sta1 = dact1[dact1.id == stat1]
    lat = sta1.lat.values[0]
    lon = sta1.lng.values[0]
    name = sta1['name'].values[0]
    loc1 = f'{name}, {sta1.state.values[0]}'
    duh.attrs = {'lat': lat, 'lon': lon, 'locat': loc1}
    return duh

def wlcoll(region_in, time_frame, outfile=None, datum='NAVD', units='metric', product='water_level'):
    """wlcoll - 'water level gauge collect', a script which collects all water level gauges
       which lie within a specified region and contain data within a specified time range.

        Parameters:
              region_in  - input a region defined either by the boundary of a grid from a netcdf grid file (so, a
                           filename / filepath), or a tuple of 4 float numbers representing lower left lat-lon and
                           upper right lat-lon, thus defining a region via a lat-lon box
              time_frame - a time frame in the form 'yyyy-mm-dd:yyyy-mm-dd' (in UTC time zone)
              outfile    - an optional, string argument that, when provided, will output the data into a netcdf file
                           with the filename given by the string
              datum      - an optional, string argument that specifies which datum to use for the requested
                           water level data [default is 'MSL', mean sea level; other options include (among others):
                           'MLLW', 'MLW', 'MHW', 'MHHW', 'DTL']

        Returns the boolean index array that yield the indices of lat/lon values that are contained within the polygon.
        Usage: e.g., dataset = wlcoll('Mexico_Beach_grd2.nc', '2023-02-24:2023-02-28', 'wl1.nc')
                or   dataset = wlcoll((24.43, -74.32, 25.76, -72.87), '2022-07-10:2022-10-18')"""

    # gather the list of active stations, and their lons / lats
    dact = pd.read_xml(
        'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml?type=waterlevels&units=english')
    lons = dact.lng[1:].values
    lats = dact.lat[1:].values
    ids = dact.id[1:].values

    # supply the region of interest and the lat-lon values of active stations to bfind
    # then extract the region-included stations
    indout = bfind(region_in, lats, lons)
    idsel = ids[indout]

    idsellen = len(idsel)
    if idsellen == 0:
        print(f'no stations found in the region in given by {region_in}')
        sys.exit(1)

    start = time_frame.split(':')[0].replace('-','')
    end = time_frame.split(':')[1].replace('-','')

    # initialize new collection of xr.DataArrays for all the buoys found
    ds1 = xr.Dataset({})
    # set a loop variable to number the gauge data arrays and their time axes
    i = 1
    # Loop through the stations found, and gather the outputs within the time frame
    # specified into the xarray Dataset ds1
    for id1 in idsel:
        station = '%d' % id1
        srchstr = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=' + product + \
                  '&application=NOS.COOPS.TAC.WL&begin_date=' + start + '&end_date=' + end + '&datum=' + datum + \
                  '&station=' + station + '&time_zone=GMT&units=' + units + '&format=csv'
        try:
            duh = pd.read_csv(srchstr)
        except:
            print(f'no data returned for station {station}')
            continue
        if product == 'water_level':
            if duh[' Water Level'].count() == 0:
                print(f'station {station} contains zero data')
                continue
        elif product == 'predictions':
            if duh[' Prediction'].count() == 0:
                print(f'station {station} contains zero data')
                continue
        lat = dact[dact.id == id1].lat.values
        lon = dact[dact.id == id1].lng.values
        name = dact[dact.id == id1].iloc[0]['name']
        loc1 = f'{name}, {dact[dact.id == id1].iloc[0].state}'
        if product == 'water_level':
            ds1['wl_' + str(i)] = xr.DataArray(data=duh[' Water Level'].values,
                                               dims=['time_' + str(i)],
                                               coords={'time_' + str(i): duh['Date Time'].values},
                                               attrs={'lat': lat, 'lon': lon, 'time': 'time_' + str(i), 'station': station, 'locat': loc1})
            ds1['sig_' + str(i)] = xr.DataArray(data=duh[' Sigma'].values,
                                                dims=['time_' + str(i)],
                                                coords={'time_' + str(i): duh['Date Time'].values},
                                                attrs={'lat': lat, 'lon': lon, 'time': 'time_' + str(i),
                                                       'station': station, 'locat': loc1})
        elif product == 'predictions':
            ds1['wl_' + str(i)] = xr.DataArray(data=duh[' Prediction'].values,
                                               dims=['time_' + str(i)],
                                               coords={'time_' + str(i): duh['Date Time'].values},
                                               attrs={'lat': lat, 'lon': lon, 'time': 'time_' + str(i),
                                                      'station': station, 'locat': loc1})
        i = i + 1

    if len(ds1) == 0:
        print(f'NO data were found for any stations in the region and time frame given.')
    elif outfile is not None:
        ds1.to_netcdf(outfile)

    return ds1

def gcdist(startlat, startlon, endlat, endlon):
    import pyproj
    g = pyproj.Geod(ellps='WGS84')
    (az12, az21, dist) = g.inv(startlon, startlat, endlon, endlat)
    return dist

def llfind(lat1,lon1,lats1,lons1, maxdist=3):
    if type(lat1) is xr.DataArray:
        lat1 = float(lat1)
    if type(lon1) is xr.DataArray:
        lon1 = float(lon1)
    if type(lats1) is xr.DataArray:
        lats1 = lats1.values
    if type(lons1) is xr.DataArray:
        lons1 = lons1.values
    lons1a = lons1[~np.isnan(lons1)]
    lats1a = lats1[~np.isnan(lats1)]
    lons1a_idx = int(lons1a.shape[0] * 0.1)
    lats1a_idx = int(lats1a.shape[0] * 0.7)
    dist1 = np.abs(lons1a[lons1a_idx] - lons1a[lons1a_idx - 1])
    dist2 = np.abs(lats1a[lats1a_idx - 1] - lats1a[lats1a_idx])
    dd = 10 * max(dist1, dist2)
    idx3 = np.where((lats1 < lat1 + dd) & (lats1 > lat1 - dd) & (lons1 < lon1 + dd) & (lons1 > lon1 - dd))
    if len(idx3[0]) < 5:
        return None
    if len(lats1.shape) == 2:
        yidxmin = idx3[0].min()
        yidxmax = idx3[0].max()
        xidxmax = idx3[1].max()
        xidxmin = idx3[1].min()
        lats2 = lats1[yidxmin:yidxmax, xidxmin:xidxmax]
        if lats2.size < 5:
            return None
        lons2 = lons1[yidxmin:yidxmax, xidxmin:xidxmax]
        if lons2.size < 5:
            return None
    elif len(lats1.shape) == 1:
        idxmin = idx3[0].min()
        idxmax = idx3[0].max()
        lats2 = lats1[idxmin:idxmax]
        if lats2.size < 5:
            return None
        lons2 = lons1[idxmin:idxmax]
        if lons2.size < 5:
            return None

    lons2a = lons2[~np.isnan(lons2)]
    lats2a = lats2[~np.isnan(lats2)]

    blah = list(zip(lats2a, lons2a))
    mp1 = MultiPoint(blah)

    p1 = Point(lat1, lon1)

    ng1 = nearest_points(p1, mp1)
    ng11 = ng1[1]

    idx = np.argwhere(((ng11.xy[0] == lats1) & (ng11.xy[1] == lons1))).squeeze()

    dist1 = gcdist(blah[0][0], blah[0][1], blah[1][0], blah[1][1])
    mididx1 = len(blah) // 2
    dist2 = gcdist(blah[mididx1][0], blah[mididx1][1], blah[mididx1 + 1][0], blah[mididx1 + 1][1])
    mindist1 = min(dist1, dist2)
    max1 = maxdist * mindist1

    if len(lats1.shape) == 2:
        if gcdist(lat1, lon1, lats1[tuple(idx)], lons1[tuple(idx)]) > max1:
            return None
        else:
            return idx
    if gcdist(lat1, lon1, lats1[idx], lons1[idx]) > max1:
        return None
    else:
        return idx

def d64(tarray):
    """takes date strings from xarray / pandas output which aren't datetime64 objects
    will normally need to send this data to the function as an np array (so using timearr.values)"""
    tout = np.ndarray(tarray.shape, dtype=object)
    for ti in range(tarray.size):
        tout[ti] = np.datetime64(tarray[ti])
    return tout

def get_vars(ds_in, varpatt, coordvars=False, return_ds=False):
    # get a boolean array indexing the locations of variables matching 'varpatt'
    if coordvars:
        idx1 = [varpatt in x for x in list(ds_in.variables.keys())]
        idxarr1 = np.array(list(ds_in.variables.keys()))[idx1]
    else:
        idx1 = [varpatt in x for x in list(ds_in)]
        idxarr1 = np.array(list(ds_in))[idx1]
    if return_ds is True:
        ds_out = ds_in[idxarr1]
        return ds_out
    else:
        return list(idxarr1)


def my_fmt_xdate(ax=None,rot=30,hal='right'):
    if ax is None:
        ax = plt.gca()

    for i in ax.get_xmajorticklabels():
        i.set_rotation(rot)
        i.set_ha(hal)

    return None

def wlqckcomp(model_file, variable='zeta', time_frame=None, interpol=None, butterord=None, datum='NAVD', product='water_level', corroff=0):
    from matplotlib import font_manager
    pltsubdir = "wlcmp/"
    if not os.path.exists(pltsubdir):
        os.mkdir(pltsubdir)
    # have to open a separate variable on the model file so I have access to mask data
    # wlmodcmp1 is great for the graphing part, but only returns the one variable in dmod1
    dmod = xr.open_dataset(model_file)
    ds1, dmod1 = wlmodcmp1(model_file, variable, time_frame=time_frame, datum=datum, product=product)

    # find all variables matching wl_* and stick them in list for iteration
    list1 = list(get_vars(ds1,'wl_'))

    if 'mask_rho' in list(dmod):
        if 'wetdry_mask_rho' in list(dmod):
            if dmod['mask_rho'].values.any():
                dwdmn = dmod.wetdry_mask_rho.min('ocean_time')
                idx2 = np.where((dmod['mask_rho'].values == 0) | (dwdmn == 0))
            else:
                idx2 = np.where(dmod1[0].isnull())
        else:
            if dmod['mask_rho'].values.any():
                idx2 = np.where(dmod['mask_rho'].values == 0)
            else:
                idx2 = np.where(dmod1[0].isnull())
    else:
        if 'wetdry_mask_rho' in list(dmod):
            dwdmn = dmod.wetdry_mask_rho.min('ocean_time')
            idx2 = np.where((dmod1[0].isnull()) | (dwdmn == 0))
        else:
            idx2 = np.where(dmod1[0].isnull())

    # find lat / lon / time variable names for 'variable' in model data
    lat_ish = ['lat' in x for x in dmod1.coords]
    modlat = dmod[np.array(list(dmod1.coords))[lat_ish][0]].values
    lon_ish = ['lon' in x for x in dmod1.coords]
    modlon = dmod[np.array(list(dmod1.coords))[lon_ish][0]].values
    time_ish = ['time' in x for x in dmod1.coords]
    modtime = dmod1[np.array(list(dmod1.coords))[time_ish][0]].values
    # now find all lat / lon coords that correspond to places with data
    # and mask them with nan so that llfind() will not look for lat-lon places within
    # the dataset that have no data
    modlon[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = np.nan
    modlat[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = np.nan

    print(f'station list is {list1}')
    prop1 = font_manager.FontProperties(size=8)
    for station in list1:
        lat1 = ds1[station].lat
        lon1 = ds1[station].lon
        time1 = ds1[ds1[station].time]
        tstr = ds1[station].time
        locat1 = ds1[station].locat
        if (lat1 < 30) & (lon1 > -71):
            continue

        tout = d64(time1.values)
        wldata = ds1[station]
        if interpol is not None:
            wli = wldata.interpolate_na(dim=tstr, method=interpol, use_coordinate=False)
            if butterord is not None:
                from scipy.signal import buttord, butter, filtfilt
                N, Wn = butterord
                #N, Wn = buttord(0.2, 0.23, 0.3, 0.6)
                b, a = butter(N, Wn)
                wlf = filtfilt(b, a, wli)
                wldata = wlf.copy()
            else:
                wldata = wli.copy()

        idx1 = llfind(lat1, lon1, modlat, modlon)
        if idx1 is None:
            continue

        moddat = dmod1[1:, idx1[0], idx1[1]]
        stdrat = wldata.std() / moddat.std()
        
        modlontmp = np.array(modlon)
        modlattmp = np.array(modlat)
        while stdrat > 10:
            modlontmp[idx1[0], idx1[1]] = np.nan
            modlattmp[idx1[0], idx1[1]] = np.nan
            idx1 = llfind(lat1, lon1, modlattmp, modlontmp)
            if idx1 is None:
                break
            moddat = dmod1[1:, idx1[0], idx1[1]]
            stdrat = wldata.std() / moddat.std()
        
        if idx1 is None:
            continue

        moddat = dmod1[:, idx1[0], idx1[1]]
        fig1 = plt.figure(figsize=(6,3))
        ax1 = fig1.add_axes((0.1,0.15,0.88,0.74))
        l1 = plt.plot(modtime, moddat, '-')
        l2 = plt.plot(tout, wldata, '-')

        modcorr = moddat[corroff:]
        wlcorr = []
        for t1 in modcorr.ocean_time:
            # get dmod's time string into same format as gauges' time strings
            cmpstr = (' '.join(str(t1.values).split('T')))[:-13]
            wlcorr.append(wldata.loc[wldata[tstr] == cmpstr].values)
        wlcorr = np.array(wlcorr).squeeze()
        if modcorr.size == wlcorr.size:
            corr1 = np.corrcoef(modcorr, wlcorr)[1, 0]
            plt.text(0.003, 0.71, f'corr:\n{corr1: .2f}', ha='left', size=7, transform=ax1.transAxes)

        ltxt1 = 'model' + f'--mean:{moddat.mean().values: .2f}'
        ltxt2 = 'obs' + f'--mean: {wldata.mean(): .2f}'
        plt.legend([ltxt1, ltxt2], loc=2, prop=prop1)

        titlstr = f'Comparison of data from {dmod.file} to obs at station {ds1[station].station}\nlocation: {locat1}  ' \
                  f'lat: {lat1}  lon: {lon1}, Datum: {datum}'
        plt.title(titlstr, size=8)

        xtl = plt.gca().get_xticklabels()
        plt.setp(xtl, 'size', 7)
        my_fmt_xdate(None, rot=30, hal='right')
        plt.ylabel('water level (m)')
        
        fileout = pltsubdir + 'wlcmp-' + ds1[station].station
        plt.savefig(fileout, dpi=200)

        plt.close()

    return ds1, dmod1

def wl1station(station, model_file, variable, time_frame, datum='NAVD'):
    duh1 = wlget(station, time_frame, datum=datum)
    lat1 = duh1.attrs['lat']
    lon1 = duh1.attrs['lon']
    time1 = duh1['Date Time'].values
    tout = d64(time1)
    locat1 = duh1.attrs['locat']

    dmod = xr.open_dataset(model_file)
    tvarname = dmod[variable].time
    lat_ish = ['lat' in x for x in dmod[variable].coords]
    modlat = dmod[np.array(list(dmod[variable].coords))[lat_ish][0]].values
    lon_ish = ['lon' in x for x in dmod[variable].coords]
    modlon = dmod[np.array(list(dmod[variable].coords))[lon_ish][0]].values
    time_ish = ['time' in x for x in dmod[variable].coords]
    modtime = dmod[np.array(list(dmod[variable].coords))[time_ish][0]].values
    # now find all lat / lon coords that correspond to places with data
    # and mask them with nan so that llfind() will not look for lat-lon places within
    # the dataset that have no data
    idx2 = np.where(dmod[variable][0].isnull())
    modlon[xr.DataArray(idx2[0]), xr.DataArray(idx2[1])] = np.nan
    modlat[xr.DataArray(idx2[0]), xr.DataArray(idx2[1])] = np.nan

    start = time_frame.split(':')[0]
    end = time_frame.split(':')[1]
    idxt = np.where((modtime > np.datetime64(start)) & (modtime < np.datetime64(end)))

    idx1 = llfind(lat1, lon1, modlat, modlon)
    if idx1 is None:
        print(f'no model data found within range of station {station}')
        return None
    wldata = duh1[' Water Level'].values
    moddat = dmod.zeta[idxt][:, idx1[0], idx1[1]]
    moddattime = modtime[idxt]

    fig1 = plt.figure(figsize=(12, 6))

    l1 = plt.plot(moddattime, moddat, '-')
    l2 = plt.plot(tout, wldata, '-')
    ltxt1 = 'model' + f' - mean:{moddat.mean().values: .2f}'
    ltxt2 = 'data' + f' - mean: {wldata.mean(): .2f}'
    plt.legend([ltxt1, ltxt2], loc=2)

    titlstr = f'Comparison of data from {dmod.file} at station {station}\nlocation: {locat1}  ' \
              f'lat: {lat1}  lon: {lon1}, Datum: {datum}'
    plt.title(titlstr, size=9)

    xtl = plt.gca().get_xticklabels()
    plt.setp(xtl, 'size', 9)
    my_fmt_xdate(None, rot=30, hal='right')
    plt.ylabel('water level (m)')

    return duh1,moddat

def wlqckcomp2(model_file, variable='zeta', time_frame=None, interpol=None, butterord=None, datum='NAVD', product='water_level', annotsz=5, ds1=None, dmod=None, corroff=0):
    import cartopy.crs as crs
    import matplotlib.cm as cm
    pltsubdir = "wlcmp/"
    if not os.path.exists(pltsubdir):
        os.mkdir(pltsubdir)
    if (ds1 is None) or (dmod is None):
        # have to open a separate variable on the model file so I have access to mask data
        # wlmodcmp1 is great for the graphing part, but only returns the one variable in dmod1
        dmod = xr.open_dataset(model_file)
        ds1, dmod1 = wlmodcmp1(model_file, variable, time_frame=time_frame, datum=datum, product=product)
    else:
        dmod1 = dmod[variable]

    # find all variables matching wl_* and stick them in list for iteration
    list1 = list(get_vars(ds1,'wl_'))

    if 'mask_rho' in list(dmod):
        if 'wetdry_mask_rho' in list(dmod):
            if dmod['mask_rho'].values.any():
                dwdmn = dmod.wetdry_mask_rho.min('ocean_time')
                idx2 = np.where((dmod['mask_rho'].values == 0) | (dwdmn == 0))
            else:
                idx2 = np.where(dmod1[0].isnull())
        else:
            if dmod['mask_rho'].values.any():
                idx2 = np.where(dmod['mask_rho'].values == 0)
            else:
                idx2 = np.where(dmod1[0].isnull())
    else:
        if 'wetdry_mask_rho' in list(dmod):
            dwdmn = dmod.wetdry_mask_rho.min('ocean_time')
            idx2 = np.where((dmod1[0].isnull()) | (dwdmn == 0))
        else:
            idx2 = np.where(dmod1[0].isnull())

    # find lat / lon / time variable names for 'variable' in model data
    lat_ish = ['lat' in x for x in dmod1.coords]
    modlat = dmod[np.array(list(dmod1.coords))[lat_ish][0]].values
    lon_ish = ['lon' in x for x in dmod1.coords]
    modlon = dmod[np.array(list(dmod1.coords))[lon_ish][0]].values
    time_ish = ['time' in x for x in dmod1.coords]
    modtime = dmod1[np.array(list(dmod1.coords))[time_ish][0]].values
    # now find all lat / lon coords that correspond to places with data
    # and mask them with nan so that llfind() will not look for lat-lon places within
    # the dataset that have no data
    modlon[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = np.nan
    modlat[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = np.nan

    ix = 0
    print(f'station list is {list1}')
    out1 = np.zeros((len(list1), 6))
    for station in list1:
        print(f'on station {station}')
        lat1 = ds1[station].lat
        lon1 = ds1[station].lon
        time1 = ds1[ds1[station].time]
        tstr = ds1[station].time
        locat1 = ds1[station].locat
        station1 = np.int32(ds1[station].station)
        if (lat1 < 30) & (lon1 > -71):
            continue
        
        idx1 = llfind(lat1, lon1, modlat, modlon)
        if idx1 is None:
            continue
        tout = d64(time1.values)
        wldata = ds1[station]
        if interpol is not None:
            wli = wldata.interpolate_na(dim=tstr, method=interpol, use_coordinate=False).values
            if butterord is not None:
                from scipy.signal import buttord, butter, filtfilt
                N, Wn = butterord
                #N, Wn = buttord(0.2, 0.23, 0.3, 0.6)
                b, a = butter(N, Wn)
                wlf = filtfilt(b, a, wli)
                wldata = wlf.copy()
            else:
                wldata = wli.copy()

        moddat = dmod1[:, idx1[0], idx1[1]]
        modcorr = moddat[corroff:]
        wlcorr = []
        for t1 in modcorr.ocean_time:
            # get dmod's time string into same format as gauges' time strings
            cmpstr = (' '.join(str(t1.values).split('T')))[:-13]
            wlcorr.append(wldata.loc[wldata[tstr] == cmpstr].values)
        wlcorr = np.array(wlcorr).squeeze()
        corr1 = np.nan
        if modcorr.size == wlcorr.size:
            corr1 = np.corrcoef(modcorr, wlcorr)[1, 0]

        out1[ix,0] = lat1
        out1[ix,1] = lon1
        out1[ix,2] = dmod1[:, idx1[0], idx1[1]].mean() - wldata.mean()
        out1[ix,3] = dmod1[:, idx1[0], idx1[1]].std(ddof=1) / wldata.std(ddof=1)
        out1[ix, 4] = corr1
        out1[ix, 5] = station1
        ix = ix + 1

    cpc = crs.PlateCarree()

    ## mean difference plot  ##########################################################
    fig1 = plt.figure(figsize=(6.4,4.8))
    ax1 = fig1.add_axes((0.08,0.05,0.92,0.92), projection=cpc)
    xlo = dmod.lon_rho.min()
    xhi = dmod.lon_rho.max()
    ylo = dmod.lat_rho.min()
    yhi = dmod.lat_rho.max()
    ax1.set_extent((xlo, xhi, ylo, yhi))
    ax1.coastlines(color='lightgray', zorder=0)

    out1 = out1[np.where(out1[:, 5] != 0)]

    c1 = ax1.scatter(out1[:, 1], out1[:, 0], c=out1[:, 2], s=30, cmap=cm.seismic, vmin=-.6, vmax=.6, zorder=3)

    cbar = plt.colorbar(c1)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)

    for ix in range(out1.shape[0]):
        ix2 = ix % 2
        if ix2 == 1:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] + 0.07, out1[ix, 0]),
                         size=annotsz, zorder=5)
        else:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] - 0.07, out1[ix, 0]),
                         size=annotsz, ha='right', zorder=5)
    titlstr = f'Model: {model_file}\nModel minus NOAA tide gauges: mean difference, datum: {datum}'
    plt.title(titlstr, size=6)
    anntxt = f"min = {out1[:,2].min(): .2f} m\nmax = {out1[:,2].max(): .2f} m\nmean = {out1[:,2].mean(): .2f} m"
    ax1.annotate(anntxt, (0.1,0.8), xycoords='axes fraction', size=8)
    gl = ax1.gridlines(crs=cpc, draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.right_labels = False
    gl.top_labels = False
    savestr = pltsubdir + "tgmap-1_mean.png"
    plt.savefig(savestr,dpi=400)

    ## std dev plot  ######################################################################
    fig1 = plt.figure(figsize=(6.4,4.8))
    ax1 = fig1.add_axes((0.08,0.05,0.92,0.92), projection=cpc)
    ax1.set_extent((xlo, xhi, ylo, yhi))
    ax1.coastlines(color='lightgray', zorder=0)

    c1=ax1.scatter(out1[:,1],out1[:,0],c=out1[:,3],s=30,cmap=cm.Spectral_r,vmin=0,vmax=2, zorder=3)
    cbar = plt.colorbar(c1)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)

    for ix in range(out1.shape[0]):
        ix2 = ix % 2
        if ix2 == 1:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] + 0.07, out1[ix, 0]),
                         size=annotsz, zorder=5)
        else:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] - 0.07, out1[ix, 0]),
                         size=annotsz, ha='right', zorder=5)
    titlstr = f'Model: {model_file}\nModel vs NOAA tide gauges: fraction of s.d., datum: {datum}'
    plt.title(titlstr, size=7, wrap=True)
    annstdtxt = f"min = {out1[:,3].min(): .2f} \nmax = {out1[:,3].max(): .2f}\nmean = {out1[:,3].mean(): .2f}"
    ax1.annotate(annstdtxt, (0.1,0.8), xycoords='axes fraction', size=9)

    gl = ax1.gridlines(crs=cpc, draw_labels=True,
                      linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.right_labels = False
    gl.top_labels = False

    savestr = pltsubdir + "tgmap-2_sd.png"
    plt.savefig(savestr, dpi=400)

    ## corr plot  #####################################################################
    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig1.add_axes((0.08, 0.05, 0.92, 0.92), projection=cpc)
    ax1.set_extent((xlo, xhi, ylo, yhi))
    ax1.coastlines(color='lightgray', zorder=0)

    c1 = ax1.scatter(out1[:, 1], out1[:, 0], c=out1[:, 4], s=30, cmap=cm.gist_rainbow, vmin=0, vmax=1, zorder=3)
    cbar = plt.colorbar(c1)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)

    for ix in range(out1.shape[0]):
        ix2 = ix % 2
        if ix2 == 1:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] + 0.07, out1[ix, 0]),
                         size=annotsz, zorder=5)
        else:
            ax1.annotate(str(int(out1[ix, 5])), (out1[ix, 1], out1[ix, 0]),
                         xytext=(out1[ix, 1] - 0.07, out1[ix, 0]),
                         size=annotsz, ha='right', zorder=5)
    titlstr = f'Model: {model_file}\nModel vs NOAA tide gauges: correlation, datum: {datum}'
    plt.title(titlstr, size=7, wrap=True)
    annstdtxt = f"min = {np.nanmin(out1[:, 4]): .2f} \nmax = {np.nanmax(out1[:, 4]): .2f}\nmedian = {np.nanmedian(out1[:, 4]): .2f}"
    ax1.annotate(annstdtxt, (0.1, 0.8), xycoords='axes fraction', size=9)

    gl = ax1.gridlines(crs=cpc, draw_labels=True,
                       linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.right_labels = False
    gl.top_labels = False

    savestr = pltsubdir + "tgmap-3_corr.png"
    plt.savefig(savestr, dpi=400)

    # plot whole grid  ##################################################################
    fig1 = plt.figure(figsize=(9,9.5))
    ax1 = fig1.add_axes((0.05,0.05,0.92,0.92), projection=cpc)
    ax1.coastlines()
    dmod = xr.open_dataset(model_file)
    pc1 = plt.pcolormesh(dmod.lon_rho,dmod.lat_rho,dmod.zeta[45])
    plt.title(f'Model domain')
    gl = ax1.gridlines(crs=cpc, draw_labels=True,
                       linewidth=2, color='gray', alpha=0, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False

    return out1

def wlmodcmp1(model_file, variable, time_frame=None, datum='NAVD', product='water_level'):
    """This function is called to simply first gather water level data (or something from the NOAA gauges, supplied
    by the 'product' parameter) that is colocated within a given model data file's grid, the model data from
    the given file, and prunes both data sets
    to the same time window.  This is either the whole time frame of the model data, which is the default, or is
    given by the 'time_frame' parameter in the form of a string of dates separated by a colon
    (i.e., time_frame='2023-02-25:2023-03-15')."""
    import re
    re_timepart = re.compile('T.*')

    # model data xr
    dmod = xr.open_dataset(model_file)

    time_ish = ['time' in x for x in dmod[variable].coords]
    modtime = dmod[np.array(list(dmod[variable].coords))[time_ish][0]]

    if time_frame is None:
        # a bit dense, but gets the first time step value (stripping it out from being a 'dataset' object),
        # subtracts a day, turns it into a string so that the T* time info can be stripped off
        begin = re_timepart.sub('', (str((modtime[0] - np.timedelta64(0, 'D')).values)))
        # the same, but for the last time point, with adding a day (so the time frame is a little bit longer than the model
        # data, just in case)
        end = re_timepart.sub('', (str((modtime[-1] + np.timedelta64(1, 'D')).values)))
        timeframe = begin + ':' + end
    else:
        start = time_frame.split(':')[0]
        end = time_frame.split(':')[1]
        idxt = np.where((modtime > np.datetime64(start)) & (modtime < np.datetime64(end)))
        timeframe = time_frame

    # get water level gauge data
    ds1 = wlcoll(model_file, timeframe, datum=datum, product=product)

    if time_frame is None:
        moddat = dmod[variable]
    else:
        moddat = dmod[variable][idxt]

    return ds1,moddat


def plot_wvht_1(buoy, dm, ds=None, save=False, ax=None):
    """Plots a single panel time series of obs NDBC buoy significant wave height
     compared to that from a model dataset, usually from ROMS (apologies if this doesn't
     work for some other netcdf dataset).
     Parameters:
         buoy - 5-digit code, in a string, for the ndbc buoy (e.g., '41004', 'sipf1')
         dm - model data, provided in an xarray dataset (i.e., no file name here, just
              open the file and provide the xarray dataset for this function)
         ds - optionally provide the dataset of gathered buoys from cdg.buoycoll(); if
              blank / None, this will look up this one buoy on-the-fly)

    E.g.,
    > import coawst_dataget as cdg
    > ds1 = cdg.buoycoll('https://geoport.whoi.edu/thredds/dodsC/vortexfs1/usgs/Projects/Michael2018/michael31/qck/michael_ocean_gomsab_qck.nc', variable='zeta')
    > dmod = xr.open_dataset('https://geoport.whoi.edu/thredds/dodsC/vortexfs1/usgs/Projects/Michael2018/michael31/qck/michael_ocean_gomsab_qck.nc')
    > cdg.plot_wvht_1('41004',dmod,ds1)
     """
    if ds is None:
        print('Not implemented yet')
        return None
    buoyvar = 'buoy_' + buoy

    lat, lon = ds[buoyvar]
    idxa, idxo = llfind(lat, lon, dm.lat_rho, dm.lon_rho)
    hw1 = dm['Hwave'][:, idxa, idxo]
    ht1 = dm[hw1.time]

    if ('wave_height_' + buoy) in list(ds):
        bd1 = ds['wave_height_' + buoy]
        timevar = 'time_' + buoy
        if ax is None:
            fig1 = plt.figure()
            ax1 = fig1.add_axes((0.12, 0.125, 0.8, 0.8))
        else:
            fig1 = ax.get_figure()
            ax1 = plt.sca(ax)
        l1 = bd1.dropna(timevar).plot(ax=ax1)
        l2 = hw1.plot(ax=ax1)

        bd1a = bd1.dropna(timevar)
        bt1 = bd1a[timevar]
        bt1a = mpl.dates.date2num(bt1)
        ht1a = mpl.dates.date2num(ht1)
        # get interpolation function
        fi1 = sp.interpolate.interp1d(bt1a, bd1a)
        # get location (time location) of interpolated values
        arr1a = np.where((ht1 < bt1[-1]) & (ht1 > bt1[0]))
        # get interpolated values of bd1
        bd2 = fi1(ht1a[arr1a])
        # Now we can calculate a proper RMSE
        rmse1 = np.sqrt((((hw1[arr1a] - bd2) ** 2) / bd2.var()).mean())
    elif ('spectral_wave_density_' + buoy) in list(ds):
        bd1 = calc_Hs_1d(ds['spectral_wave_density_' + buoy],ds['frequency_' + buoy])
        timevar = 'time_' + buoy + '_sw'
        if ax is None:
            fig1 = plt.figure()
            ax1 = fig1.add_axes((0.12, 0.125, 0.8, 0.8))
        else:
            fig1 = ax.get_figure()
            ax1 = plt.sca(ax)

        bt1 = ds[timevar]
        bt1a = mpl.dates.date2num(bt1)
        ht1a = mpl.dates.date2num(ht1)

        l1 = plt.plot(bt1,bd1)
        l2 = hw1.plot(ax=ax1)

        # get interpolation function
        fi1 = sp.interpolate.interp1d(bt1a, bd1)
        # get location (time location) of interpolated values
        arr1a = np.where((ht1 < bt1[-1]) & (ht1 > bt1[0]))
        # get interpolated values of bd1
        bd2 = fi1(ht1a[arr1a])
        # Now we can calculate a proper RMSE
        rmse1 = np.sqrt((((hw1[arr1a] - bd2) ** 2) / bd2.var()).mean())
    else:
        print(f'No wave data for buoy {buoy}')
        return None
    modelstr = f'model, rmse = {rmse1:0.2f}'
    plt.legend(['obs', modelstr], loc=2)
    titlestr = "Hs @ " + ds[buoyvar].comment + f' ({buoy})'
    plt.title(titlestr, size=10)

    if save is True:
        if not os.path.exists('Hs'):
            os.mkdir('Hs')
        savestr = 'Hs/wvht_' + buoy
        plt.savefig(savestr, dpi=100)
        plt.close()
        return None
    return fig1, ax1


def to_moment(R1, R2, alpha1, alpha2):
    """
    Convert NDBC directional parameters r1, r2, theta1, theta2 to Fourier coeffs. a1, a2, b1, and b2.

    This code was written by Pieter Smit and provided courtesy of Isabel Houghton, Sofar Ocean.

    Note: from the NDBC website

    The R1 and R2 values in the monthly and yearly historical data files are scaled by 100,
    a carryover from how the data are transported to the archive centers. The units are
    hundredths, so the R1 and R2 values in those files should be multiplied by 0.01.

    May have to multiply R1/R2 with a 100 depending on source

    :param R1:
    :param R2:
    :param alpha1:
    :param alpha2:
    :return:
    """
    to_rad = np.pi / 180
    eps = 1e-16
    angle1 = (270 - alpha1) * to_rad
    angle2 = (540 - 2 * alpha2) * to_rad

    a0 = 1 / (2 * np.pi) + eps
    a1 = R1 * np.cos(angle1) / a0
    b1 = R1 * np.sin(angle1) / a0
    a2 = R2 * np.cos(angle2) / a0
    b2 = R2 * np.sin(angle2) / a0

    return a1, b1, a2, b2


def calc_theta_mean_a1b1(a1, b1, freqvar):
    # mean direction (degrees)
    # no weighting
    # Le Merle et al., 2021 eqn 5; Isabels code, SoFar manual
    a1bar = np.ndarray(a1.shape[0])
    b1bar = np.ndarray(a1.shape[0])
    for t1 in range(a1.shape[0]):
        a1bar[t1] = np.trapz(a1[t1].dropna(freqvar), a1[t1].dropna(freqvar)[freqvar])
        b1bar[t1] = np.trapz(b1[t1].dropna(freqvar), b1[t1].dropna(freqvar)[freqvar])
    thetam = 270. - (180. / np.pi) * np.arctan2(b1bar, a1bar)
    return thetam


def plot_wvdir_1(buoy, dm, ds=None, save=False, ax=None):
    """Plots a single panel time series of obs NDBC buoy mean wave direction
     compared to that from a model dataset, usually from ROMS (apologies if this doesn't
     work for some other netcdf dataset).
     Parameters:
         buoy - 5-digit code, in a string, for the ndbc buoy (e.g., '41004', 'sipf1')
         dm - model data, provided in an xarray dataset (i.e., no file name here, just
              open the file and provide the xarray dataset for this function)
         ds - optionally provide the dataset of gathered buoys from cdg.buoycoll(); if
              blank / None, this will look up this one buoy on-the-fly)
    E.g.,
    > import coawst_dataget as cdg
    > ds1 = cdg.buoycoll('https://geoport.whoi.edu/thredds/dodsC/vortexfs1/usgs/Projects/Michael2018/michael31/qck/michael_ocean_gomsab_qck.nc', variable='zeta')
    > dmod = xr.open_dataset('https://geoport.whoi.edu/thredds/dodsC/vortexfs1/usgs/Projects/Michael2018/michael31/qck/michael_ocean_gomsab_qck.nc')
    > plot_wvdir_1('41004',dmod,ds1)
    """
    buoyvar = 'buoy_' + buoy
    timevar = 'time_' + buoy + '_sw'
    freqvar = 'frequency_' + buoy
    pdirvar = 'principal_wave_dir_' + buoy
    mdirvar = 'mean_wave_dir_' + buoy
    r1var = 'wave_spectrum_r1_' + buoy
    r2var = 'wave_spectrum_r2_' + buoy
    lat, lon = ds[buoyvar]
    idxa, idxo = cdg.llfind(lat, lon, dm.lat_rho, dm.lon_rho)
    dw1 = dm['Dwave'][:, idxa, idxo]
    dwt1 = dm[dw1.time]
    mc1 = to_moment(ds[r1var], ds1[r2var],
                    ds[mdirvar], ds[pdirvar])
    bd1 = calc_theta_mean_a1b1(mc1[0], mc1[1], freqvar)
    bd2 = np.where(bd1 > 360, bd1 - 360, bd1)

    if ax is None:
        fig1 = plt.figure()
        ax1 = fig1.add_axes((0.12, 0.125, 0.8, 0.8))
    else:
        fig1 = ax.get_figure()
        ax1 = plt.sca(ax)
    l1 = plt.plot(mc1[0][timevar], bd2)
    l2 = dw1.plot(ax=ax1)

    # modelstr = f'model, rmse = {rmse1:0.2f}'
    modelstr = 'model'
    plt.legend(['obs', modelstr], loc=2)
    titlestr = "Mean wave dir @ " + ds[buoyvar].comment + f' ({buoy})'
    plt.title(titlestr, size=9)

    if save is True:
        if not os.path.exists('wvdir_out'):
            os.mkdir('wvdir_out')
        savestr = 'wvdir_out/wvdir_' + buoy
        plt.savefig(savestr, dpi=100)
        plt.close()
        return None
    return fig1, ax1


def plot_wvpp_1(buoy, dm, ds=None, save=False, ax=None):
    """Plots peak wave period comparison between model and NDBC buoy data"""
    buoyvar = 'buoy_' + buoy
    timevar = 'time_' + buoy + '_sw'
    freqvar = 'frequency_' + buoy
    swdenvar = 'spectral_wave_density_' + buoy
    lat, lon = ds[buoyvar]
    idxa, idxo = cdg.llfind(lat, lon, dm.lat_rho, dm.lon_rho)
    pw1 = dm['Pwave_top'][:, idxa, idxo]
    pwt1 = dm[pw1.time]
    bd1 = calc_peak_T(ds[swdenvar], ds[freqvar])

    if ax is None:
        fig1 = plt.figure()
        ax1 = fig1.add_axes((0.12, 0.125, 0.8, 0.8))
    else:
        fig1 = ax.get_figure()
        ax1 = plt.sca(ax)
    l1 = plt.plot(ds[timevar], bd1)
    l2 = pw1.plot(ax=ax1)

    # modelstr = f'model, rmse = {rmse1:0.2f}'
    modelstr = 'model'
    plt.legend(['obs', modelstr], loc=2)
    titlestr = "Peak wave period @ " + ds[buoyvar].comment + f' ({buoy})'
    plt.title(titlestr, size=9)

    if save is True:
        if not os.path.exists('Pwave'):
            os.mkdir('Pwave')
        savestr = 'Pwave/wvdir_' + buoy
        plt.savefig(savestr, dpi=100)
        plt.close()
        return None
    return fig1, ax1


def calc_Qp_1d(spec1d, frequencies):
    # frequency peakedness () Le Merle et al., 2021 eqn 4 based on Goda (1976)
    return 2. * np.trapz(spec1d ** 2, frequencies) / (np.trapz(spec1d, frequencies)) ** 2


def calc_m0_1d(spec1d, frequencies):
    return np.trapz(spec1d, frequencies)


def calc_m1_1d(spec1d, frequencies):
    return np.trapz(spec1d * frequencies, frequencies)


def calc_peak_T(spec1d, frequencies):
    freqout = np.ndarray(spec1d.shape[0])
    for f1 in range(freqout.size):
        maxfilt = spec1d[f1] == spec1d[f1].max()
        freqout[f1] = frequencies[maxfilt]
    return 1.0 / freqout

def specplt(Sfa, dirs=None, freqs=None, cmap=cm.Spectral_r, ax = None, cb=True, rticks=True):
    """Sfa = spectral density function as a function of freq and dir
    dirs = directions for Sfa data, can be omitted if Sfa contains this info (xarray)
    freqs = same as dirs but are the frequencies
    cmap = colormap used to plot the spectral intensity
    """
    if ax is None:
        ax = plt.subplot(projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_rscale('log')
    lev1a = np.array([0.01])
    lev1b = np.arange(0.1, 0.55, 0.1)
    lev1 = np.append(lev1a, lev1b)
    if dirs is None:
        dirs = Sfa.dir
    if freqs is None:
        freqs = Sfa.freq
    cf1 = ax.contourf(np.deg2rad(dirs), freqs, Sfa, cmap=cm.Spectral_r, vmin=0.05, levels=lev1, extend='max')
    ax.set_ylim((0.05, 0.4))
    ax.set_yscale('log')
    ax.set_rticks([0.04, 0.05, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(['', '45', '', '135', '', '225', '', '315'])
    ax.set_yticklabels(['', '0.05', '0.1', '0.2', '0.3', '0.4 Hz'], fontdict={'fontsize': 7})
    ax.yaxis.set_minor_formatter('')
    if cb:
        plt.colorbar(cf1)
    return cf1


def my_efth(ds1):
    import numpy as np
    from scipy.special import cosdg

    degperrad = 180 / np.pi
    dirs1 = ds1['mean_wave_dir']
    specs1 = ds1['spectral_wave_density']
    r1 = ds1.wave_spectrum_r1
    r2 = ds1.wave_spectrum_r2
    pdirs1 = ds1.principal_wave_dir
    freqs1 = ds1.frequency

    shp1 = r1.shape
    shp1a = np.append(shp1, 1)
    alph = np.arange(0, 365, 10)

    Dfa = (0.5 + r1.values.reshape(shp1a) * cosdg(
        alph.reshape(1, 1, alph.size) - dirs1.values.reshape(shp1a)) +
           r2.values.reshape(shp1a) * cosdg(
                2 * (alph.reshape(1, 1, alph.size) - pdirs1.values.reshape(shp1a)))) / np.pi

    Sfa = specs1.values.reshape(shp1a) * Dfa
    Sfa1 = np.ma.masked_invalid(Sfa).squeeze()
    Sfa1 = Sfa1 / degperrad
    dsout = xr.Dataset()
    ds1 = ds1.rename({'frequency': 'freq'})
    dsout['efth'] = xr.DataArray(data=Sfa1,
                                dims=['time', 'freq', 'dir'],
                                coords={'time' : ds1.time, 'freq': ds1.freq, 'dir': alph})

    return dsout

def ndbcspectxt(buoy,year):
    import glob
    import wavespectra as ws
    f_r1 = glob.glob(buoy + 'j' + year + '*.txt')[0]
    f_r2 = glob.glob(buoy + 'k' + year + '*.txt')[0]
    f_a1 = glob.glob(buoy + 'd' + year + '*.txt')[0]
    f_a2 = glob.glob(buoy + 'i' + year + '*.txt')[0]
    f_swd = glob.glob(buoy + 'w' + year + '*.txt')[0]
    ds1 = xr.Dataset()
    w1_r1 = ws.read_ndbc_ascii(f_r1)
    if w1_r1.efth[0][0].squeeze() > 10:
        ds1['wave_spectrum_r1'] = w1_r1.efth / 100.
    else:
        ds1['wave_spectrum_r1'] = w1_r1.efth
    w1_r2 = ws.read_ndbc_ascii(f_r2)
    if w1_r2.efth[0][0].squeeze() > 10:
        ds1['wave_spectrum_r2'] = w1_r2.efth / 100.
    else:
        ds1['wave_spectrum_r2'] = w1_r2.efth

    w1_a1 = ws.read_ndbc_ascii(f_a1)
    ds1['mean_wave_dir'] = w1_a1.efth
    w1_a2 = ws.read_ndbc_ascii(f_a2)
    ds1['principal_wave_dir'] = w1_a2.efth
    w1_swd = ws.read_ndbc_ascii(f_swd)
    ds1['spectral_wave_density'] = w1_swd.efth

    ds1 = ds1.squeeze()
    ds1 = ds1.rename({'freq': 'frequency'})
    return ds1

def integrate_in_frequency(data_field, frequencies, faxis=0):
    return np.trapz(data_field, frequencies, axis=faxis).squeeze()


def integrate_in_direction(data_field, directional_bin_width, daxis=1):
    return np.sum(data_field * directional_bin_width, axis=daxis).squeeze()


def integrate_frequency_and_direction(data_field, frequencies, directional_bin_width,
                                      faxis=0, daxis=1):
    fsum = integrate_in_frequency(data_field, frequencies, faxis=faxis).squeeze()
    return integrate_in_direction(fsum, directional_bin_width, daxis=daxis).squeeze()


def calc_Hs_2d( data_field, frequencies, directional_bin_width, faxis=0, daxis=1 ):
    return( 4.*np.sqrt( integrate_frequency_and_direction(data_field, frequencies,
                        directional_bin_width, faxis=faxis, daxis=daxis).squeeze()) )

def calc_Hs_1d(data_field, frequencies):
    m0 = calc_m0_1d(data_field, frequencies)
    return(4.*np.sqrt(m0))

