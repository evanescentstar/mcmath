# Imports
import pandas as pd
import xarray as xr
import numpy as np
from shapely import geometry
import requests
import cartopy.io.shapereader as shpreader
import re
import sys
import os
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

def poly_region(region_in):
    """Returns a shapely polygon defined either by the boundary of a grid from a netcdf grid file (so, a
    filename / filepath) or a tuple of 4 float numbers representing lower left lat-lon and
    upper right lat-lon, thus defining the polygon of a lat-lon box."""

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
        grdH = grd.h.load()
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
        y1, x1, y2, x2 = boundbox
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


def ndbcget(buoy, btype, timeframe):
    """Grabs the buoy data from the dods.ndbc.noaa.gov THREDDS webserver for buoy with id 'buoy'
    and type 'btype' (one of 'stdmet', 'swden', 'ocean', 'pwind', or 'cwind', and within the time frame
    given by a tuple of date strings, such as ('2022-03-15', '2022-04-29'), or through to the latest data,
    ('2023-03-15', 'latest') - in UTC time zone.

    E.g., ndbcget('41112', 'stdmet', ('2022-03-15', '2022-04-29'))
    """
    import datetime
    from datetime import datetime as dt
    print(f'looking up buoy {buoy} in {btype}, over the dates {timeframe[0]} to {timeframe[1]}')
    start = datestrnorm(timeframe[0])
    if timeframe[1] in ['latest', 'present', 'today']:
        end = (dt.today() + datetime.timedelta(1)).strftime('%Y-%m-%d 23:59:59')
    else:
        end = datestrnorm(timeframe[1])
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
        print(f"ERROR: buoy type '{btype}' not found")
        print(f"must be one of: {varset.keys()}")
        return None
    prefdata = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/' + btype + '/'
    prefhtml = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/' + btype + '/'
    dir1 = prefhtml + buoy + suff
    ddir1 = pd.read_html(dir1)[0]
    if len(ddir1.Dataset[ddir1.Dataset.str.contains('9999')]) != 0:
        print(f'buoy {buoy} in {btype}')
        ds1 = xr.Dataset({})
        dset = ddir1.Dataset[ddir1.Dataset.str.contains('9999')]
        dset1 = prefdata + buoy + '/' + dset.values[0]
        ddset1 = xr.open_dataset(dset1)
        timefr = (ddset1.time <= np.datetime64(end)) & (ddset1.time >= np.datetime64(start))
        if timefr.values.sum() == 0:
            print(f'buoy {buoy} has no data within this time frame')
        else:
            timecoord = ddset1.time[timefr].values
            if btype == 'swden':
                freqcoord = ddset1.frequency.values
            for v1 in varset[btype]:
                vout = ddset1.variables[v1][timefr].squeeze()
                if np.count_nonzero(~np.isnan(vout.values)) == 0:
                    print(f'var {v1} is empty, skipping')
                    continue
                elif btype == 'swden':
                    ds1[v1] = xr.DataArray(data=vout.values,
                                           dims=list(vout.dims),
                                           coords={'time': timecoord, 'frequency': freqcoord},
                                           attrs=vout.attrs)
                else:
                    ds1[v1] = xr.DataArray(data=vout.values,
                                           dims=list(vout.dims),
                                           coords={'time': timecoord},
                                           attrs=vout.attrs)
    if (ds1 is not None):
        if len(list(ds1.variables)) > 0:
            ds1.attrs = ddset1.attrs
            ds1['latitude'] = xr.DataArray(data=ddset1.latitude.values, dims=['latitude'], attrs=ddset1.latitude.attrs)
            ds1['longitude'] = xr.DataArray(data=ddset1.longitude.values, dims=['longitude'],
                                            attrs=ddset1.longitude.attrs)
        else:
            print(f'no variables with data found for buoy {buoy} in {btype}')
            return None
    return ds1
## END of ndbcget()


def buoycoll(region_in, time_frame, outfile=None):
    """buoycoll - 'buoy collect', a script which collects all NDBC buoys which lie within a specified region
    and contain data within a specified time range.

    Parameters:
          region_in  - input a region defined either by the boundary of a grid from a netcdf grid file (so, a
                       filename / filepath), or a tuple of 4 float numbers representing lower left lat-lon and
                       upper right lat-lon, thus defining a region via a lat-lon box
          time_frame - a time frame in the form 'yyyy-mm-dd:yyyy-mm-dd' (in UTC time zone)
          outfile    - an optional argument that, when provided as a string, will output the data into a netcdf file
                       with the filename given by the string

    Returns the boolean index array that yield the indices of lat/lon values that are contained within the polygon.
    Usage: e.g., dataset = buoycoll('Mexico_Beach_grd2.nc', '2023-02-24:2023-02-28', 'buoys1.nc')
            or   dataset = buoycoll((24.43, -74.32, 25.76, -72.87), '2022-07-10:2022-10-18')"""

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

    start = time_frame.split(':')[0]
    end = time_frame.split(':')[1]

    print(buoy1)
    print(f'start: {start}')
    print(f'end: {end}')

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
            dout = ndbcget(buoy, 'stdmet', (start, end))
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
            dout = ndbcget(buoy, 'ocean', (start, end))
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
            dout = ndbcget(buoy, 'swden', (start, end))
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
            dout = ndbcget(buoy, 'pwind', (start, end))
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
            dout = ndbcget(buoy, 'cwind', (start, end))
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


def wlcoll(region_in, time_frame, outfile=None, datum='MSL', units='metric'):
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
    stations = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml?type=waterlevels&units=english'
    r = requests.get(stations)
    b1 = r.content.decode()
    dact = pd.read_xml(b1)
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
        srchstr = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=NOS.COOPS.TAC.WL' \
                  '&begin_date=' + start + '&end_date=' + end + '&datum=' + datum + '&station=' + station + \
                  '&time_zone=GMT&units=' + units + '&format=csv'
        try:
            duh = pd.read_csv(srchstr)
        except:
            print(f'no data returned for station {station}')
            continue
        if duh[' Water Level'].count() == 0:
            continue
        lat = dact[dact.id == id1].lat.values
        lon = dact[dact.id == id1].lng.values
        ds1['wl_' + str(i)] = xr.DataArray(data=duh[' Water Level'].values,
                                           dims=['time_' + str(i)],
                                           coords={'time_' + str(i): duh['Date Time'].values},
                                           attrs={'lat': lat, 'lon': lon, 'time': 'time_' + str(i), 'station': station})
        ds1['sig_' + str(i)] = xr.DataArray(data=duh[' Sigma'].values,
                                            dims=['time_' + str(i)],
                                            coords={'time_' + str(i): duh['Date Time'].values},
                                            attrs={'lat': lat, 'lon': lon, 'time': 'time_' + str(i),
                                                   'station': station})
        i = i + 1

    if len(ds1) == 0:
        print(f'NO data were found for any stations in the region and time frame given.')
    elif outfile is not None:
        ds1.to_netcdf(outfile)

    return ds1

def llfind(lat1,lon1,lats1,lons1):
    lons1a = lons1[~np.isnan(lons1)]
    lats1a = lats1[~np.isnan(lats1)]
    blah = list(zip(lats1a, lons1a))
    mp1 = MultiPoint(blah)

    p1 = Point(lat1,lon1)

    ng1 = nearest_points(p1, mp1)
    ng11 = ng1[1]

    idx = np.argwhere(((ng11.xy[0] == lats1) & (ng11.xy[1] == lons1)))
    return idx.squeeze()

def d64(tarray):
    """takes date strings from xarray / pandas output which aren't datetime64 objects
    will normally need to send this data to the function as an np array (so using timearr.values)"""
    tout = np.ndarray(tarray.shape, dtype=object)
    for ti in range(tarray.size):
        tout[ti] = np.datetime64(tarray[ti])
    return tout

def get_vars(ds_in, varpatt, coordvars=False):
    # get a boolean array indexing the locations of variables matching 'varpatt'
    if coordvars:
        idx1 = [varpatt in x for x in list(ds_in.variables.keys())]
        idxarr1 = np.array(list(ds_in.variables.keys()))[idx1]
    else:
        idx1 = [varpatt in x for x in list(ds_in)]
        idxarr1 = np.array(list(ds_in))[idx1]
    ds_out = ds_in[idxarr1]
    return ds_out


def my_fmt_xdate(ax=None,rot=30,hal='right'):
    if ax is None:
        ax = plt.gca()

    for i in ax.get_xmajorticklabels():
        i.set_rotation(rot)
        i.set_ha(hal)

    return None

def wlqckcomp(model_file, variable, interpol='linear', butterord=None, noplots=False):
    import re
    re_timepart = re.compile('T.*')

    # model data xr
    dmod = xr.open_dataset(model_file)
    tvarname = dmod[variable].time
    # a bit dense, but gets the first time step value (stripping it out from being a 'dataset' object),
    # subtracts a day, turns it into a string so that the T* time info can be stripped off
    begin = re_timepart.sub('',(str(dmod[tvarname][0].values - np.timedelta64(0, 'D'))))
    # the same, but for the last time point, with adding a day (so the time frame is a little bit longer than the model
    # data, just in case)
    end = re_timepart.sub('', (str(dmod[tvarname][-1].values + np.timedelta64(0, 'D'))))

    # get water level gauge data
    ds1 = wlcoll(model_file,begin + ':' + end)

    # find all variables matching wl_* and stick them in list for iteration
    #buh1 = ['wl_' in x for x in ds1]
    list1 = list(get_vars(ds1,'wl_'))

    # find lat / lon variable names for 'variable' in model data
    lat_ish = ['lat' in x for x in dmod[variable].coords]
    modlat = dmod[np.array(list(dmod[variable].coords))[lat_ish][0]].values
    lon_ish = ['lon' in x for x in dmod[variable].coords]
    modlon = dmod[np.array(list(dmod[variable].coords))[lon_ish][0]].values
    # now find all lat / lon coords that correspond to places with data
    # and mask them with nan so that llfind() will not look for lat-lon places within
    # the dataset that have no data
    idx2 = where(dmod[variable][0].isnull())
    modlon[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = nan
    modlat[xr.DataArray(idx2[0]),xr.DataArray(idx2[1])] = nan

    modtime = dmod[tvarname].values

    ix = 1
    print(f'station list is {list1}')
    for station in list1:
        lat1 = ds1[station].lat
        lon1 = ds1[station].lon
        time1 = ds1[ds1[station].time]
        tstr = ds1[station].time

        idx1 = llfind(lat1, lon1, modlat, modlon)
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

        figure(figsize=(12,5))
        l1 = plot(modtime, dmod.zeta[:, idx1[0], idx1[1]], '-')
        l2 = plot(tout, wldata, '-')

        legend(['model', 'data'], loc=2)
        titlstr = f'Comparison of data from {dmod.file} at station {ds1[station].station}\nlat: {lat1}  lon: {lon1}'
        title(titlstr, size=9)

        xtl = plt.gca().get_xticklabels()
        plt.setp(xtl, 'size', 9)
        my_fmt_xdate(None, rot=30, hal='right')

        fileout = 'plot-out/wlcmp-' + ds1[station].station
        plt.savefig(fileout, dpi=200)

        plt.close()

    return None