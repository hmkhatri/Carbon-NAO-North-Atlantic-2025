"""
This script can be used to compute 
1. Climatology of mixed layer depth.
2. Climatology of vertical structure of temperatures and DIC.
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

### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)
    ds = xmip.correct_lon(ds)

    return ds

def area_mean(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-sums
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_mean : timeseris of spatially-averaged dataset
    """
    
    ds_mean = (ds * dA).sum([x, y]) / dA.sum([x, y])
    
    return ds_mean

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/"

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

var_list = ['thetao', 'dissic', 'wo', 'mlotst']

# read grid cell areas and compute grid cell thicknesses on tracer grid
ds_area = xr.open_mfdataset(cmip_dir +  "../piControl/r1i1p1f2/Ofx/areacello/gn/latest/*nc", chunks=-1)
ds_area = ds_area.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)

ds_area = ds_area.compute()        
ds_area = xmip_wrapper(ds_area)

dir_list = glob.glob(cmip_dir + "/r*")

ds_save = xr.Dataset() # dataset to save output

for dir1 in dir_list[0:1]:
    
    dir_name = dir1.split('/')[-1].split(',')[0]
    
    for var in var_list:
        
        var_path = cmip_dir + dir_name + "/Omon/" + var + "/gn/latest/" + var + "*nc"
        ds = xr.open_mfdataset(var_path, chunks={'time':10})
        ds = ds.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
        ds = xmip_wrapper(ds)

        if(var == 'wo'):
            ds = ds.rename({'lev':'lev_w'})
        ds = xr.merge([ds, ds_area['areacello']])
        
        cell_area = xr.ones_like(ds[var].isel(time=0)) * ds['areacello']
        cell_area = cell_area * (ds[var].isel(time=0) / ds[var].isel(time=0)).fillna(0.) # remove values for in-land grid cells

        ## ------- Spatial averages -----------
        # 1. Subpolar North Atlantic
        dA = cell_area.where((ds['lat']>=45.) & (ds['lat']<=60.) & (ds['lon']>=310.) & (ds['lon']<=340.))
        var_area_int = area_mean(ds[var], dA = dA, x='x', y='y')
        ds_save[var + '_North_Atlantic_Subpolar'] = var_area_int.compute()
        
        # 2. Mid-lat North Atlantic
        dA = cell_area.where((ds['lat']>=25.) & (ds['lat']<=45.) & (ds['lon']>=300.) & (ds['lon']<=330.))
        var_area_int = area_mean(ds[var], dA = dA, x='x', y='y')
        ds_save[var + '_North_Atlantic_Midlat'] = var_area_int.compute()
        
        # 3. Sub-Tropical North Atlantic
        dA = cell_area.where((ds['lat']>=10.) & (ds['lat']<=25.) & (ds['lon']>=310.) & (ds['lon']<=340.))
        var_area_int = area_mean(ds[var], dA = dA, x='x', y='y')
        ds_save[var + '_North_Atlantic_Subtropical'] = var_area_int.compute()

    ds_save = ds_save.groupby('time.month').mean('time').compute() # compute climatology and save data
    save_file_path = (save_path + "/all_ensembles/Climatology/MLD_thetao_dissic.nc")
    ds_save.to_netcdf(save_file_path)
            
    print("Data saved succefully")



