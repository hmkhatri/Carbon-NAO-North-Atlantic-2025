"""
The script is set up for computing tracer times-series using CMIP6 models outputs.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os
import numpy as np
import scipy.stats as sc
import xmip.preprocessing as xmip

import warnings
warnings.filterwarnings("ignore")

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()

### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)
    ds = xmip.correct_lon(ds)

    return ds

def area_sum(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-sums
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_mean : timeseris of spatially-integrated dataset
    """
    
    ds_mean = (ds * dA).sum([x, y])
    
    return ds_mean

def Moving_Avg(ds, time = 1., time_len = 12):
    
    """Compute moving averages
    Parameters
    ----------
    ds : xarray Dataset for data variables
    time : time values for computing weights
    time_len : number of grid points for moving avg
    
    Returns
    -------
    ds_avg : Dataset containting moving avg
    """
    
    if(len(time) == 1):
        
        ds_avg = ds.rolling(time = time_len, center = True).mean('time')
        
    else: 
    
        days = time.dt.daysinmonth
        
        ds_avg = ((ds * days).rolling(time = time_len, center = True).mean('time') /
                  days.rolling(time = time_len, center = True).mean('time'))
    
    return ds_avg

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

#var_list = ['dissic', 'mlotst']
#var_list = ['epc100', 'epp100', 'expc', 'no3']
#var_list = ['epc100', 'epp100', 'expc', 'detoc', 'co3', 'phyc']
var_list = ['epcalc100']

dir_list = glob.glob(save_path + "r*")

ds_area = xr.open_mfdataset(cmip_dir + "piControl/r1i1p1f2/Ofx/areacello/gn/latest/*nc", chunks=-1)
ds_area = ds_area.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
ds_area = xmip_wrapper(ds_area)
ds_area = ds_area.compute()

for dir1 in dir_list:

    print("Ensemble member running: ", dir1)

    dir_name = dir1.split('/')[-1].split(',')[0]
    
    for var1 in var_list:

        # read tracer
        d1 = xr.open_mfdataset(cmip_dir +  "historical/" + dir_name + "/Omon/" + 
                               var1 + "/gn/latest/" + var1 + "*nc", 
                               chunks={'time':20}, use_cftime=True)
        d1 = xmip_wrapper(d1)
        d1 = d1.isel(x=slice(180, 320), y=slice(180, 320))
        

        # merge data in one dataset
        ds = xr.merge([d1.get([var1, 'lat', 'lon']), ds_area.get(['areacello'])])
        #print("Data reading completed")
        
        # Perform area-integration
        #cell_area = ds['areacello']
        
        cell_area = xr.ones_like(ds[var1].isel(time=0)) * ds['areacello']
        cell_area = cell_area * (ds[var1].isel(time=0) / ds[var1].isel(time=0)).fillna(0.) # remove values for in-land grid cells
        cell_area = cell_area.compute()
        
        ds_save = xr.Dataset()
        #if(var1 != 'mlotst'):
        #    ds_save['dz'] = (d1['lev_bounds'].isel(time=0,bnds=1) 
        #                    - d1['lev_bounds'].isel(time=0,bnds=0)).drop('time')


        # 1. Subpolar North Atlantic
        dA = cell_area.where((ds['lat']>=48.) & (ds['lat']<=65.) & (ds['lon']>=295.) & (ds['lon']<=340.))
        var_area_int = area_sum(ds[var1], dA = dA, x='x', y='y')
        if(var1 == 'mlotst'):
            ds_save[var1 + '_North_Atlantic_Subpolar'] = (var_area_int / dA.sum(['x', 'y'])).compute()
        else:
            ds_save[var1 + '_North_Atlantic_Subpolar'] = var_area_int.compute()
            
        # 2. Sub-Tropical North Atlantic
        dA = cell_area.where((ds['lat']>=25.) & (ds['lat']<=47.) & (ds['lon']>=280.) & (ds['lon']<=340.))
        var_area_int = area_sum(ds[var1], dA = dA, x='x', y='y')
        if(var1 == 'mlotst'):
            ds_save[var1 + '_North_Atlantic_Subtropical'] = (var_area_int / dA.sum(['x', 'y'])).compute()
        else:
            ds_save[var1 + '_North_Atlantic_Subtropical'] = var_area_int.compute()

        # Save data
        save_file_path = (save_path + dir_name + 
                            "/Timeseries/" + var1 + ".nc")
        #ds_save = ds_save.compute()
        ds_save.to_netcdf(save_file_path)
            
        print("Data saved succefully - ", var1)
            
        ds_save.close()
        ds.close()
        