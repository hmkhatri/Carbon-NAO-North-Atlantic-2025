"""
This script can be used to compute domain-averaged surface variables and air-sea fluxes using cmip6 simulations. 
Currently, the script is set up for the whole North Atlantic.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
import os

import warnings
warnings.filterwarnings("ignore")

#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client
#client = Client()

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
    ds_mean : timeseris of spatially-integrated dataset
    """
    
    ds_mean = (ds * dA).sum(dim = [x, y], skipna=True) / dA.sum(dim = [x, y], skipna=True)
    
    return ds_mean

def detrend_dim(da, da_ref, dim, deg=1):
    # detrend along a single dimension
    p = da_ref.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, da_ref, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, da_ref, dim, deg=deg)
    return da_detrended

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"
save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

var_list = ['thetao', 'dissic', 'hfds', 'fgco2']

for exp in experiment_id:
    for model in source_id:

        # Read picontrol simulation data
        ds_area = xr.open_mfdataset(cmip_dir + model + 
                                    "piControl/r1i1p1f2/Ofx/areacello/gn/latest/*nc", chunks=-1)
        ds_area = xmip_wrapper(ds_area)
        ds_area = ds_area.isel(x=slice(180, 310), y=slice(130,300)) # only North Atlantic region

        ds_ctr = []
        for var in var_list:
            d = xr.open_mfdataset(cmip_dir + model + "piControl/r1i1p1f2/Omon/" + var + 
                                  "/gn/latest/" + var + "*nc", chunks={'time':100})
            ds_ctr.append(d[var])
        ds_ctr = xr.merge(ds_ctr)
        ds_ctr = xmip_wrapper(ds_ctr)
        ds_ctr = ds_ctr.isel(lev=0, x=slice(180, 310), y=slice(130,300)) # only North Atlantic region

        # Read Historical simulations
        dir_list = glob.glob(cmip_dir + model + exp + "/r*")
        
        for dir1 in dir_list[7:]:
            print("Ensemble member running: ", dir1)
            dir_name = dir1.split('/')[-1].split(',')[0]
            
            ds = []
            for var in var_list:
                d = xr.open_mfdataset(cmip_dir + model + exp +  "/" + dir_name + "/Omon/" + var + 
                                  "/gn/latest/" + var + "*nc", chunks={'time':100})
                ds.append(d[var])
            ds = xr.merge(ds)
            ds = xmip_wrapper(ds)
            ds = ds.isel(lev=0, x=slice(180, 310), y=slice(130,300)) # only North Atlantic region
            ds = ds.compute()

            # read branching time of historical run from picontrol run
            hist_branch_day = d.attrs['branch_time_in_parent']
            print("Branch_parent = ", hist_branch_day)
            
            tim1 = ds_ctr['time']
            day_count = tim1.dt.days_in_month.cumsum('time')
            day_count = day_count.where(day_count < hist_branch_day)
            start_idx = (day_count / day_count).sum('time') #count nonnan-points, which are days before branching out
            
            ds_ref = ds_ctr.isel(time=slice(int(start_idx.values), int(start_idx.values) +
                                            len(ds['time']))).drop('time')
            ds_ref = ds_ref.assign_coords(time=ds['time']).compute()
                
            # Perform area-mean
            cell_area = ds_area['areacello'].where((ds['thetao'].isel(time=0) > -1.e6) & 
                                                   (ds['thetao'].isel(time=0) < 1.e6))
            dA = cell_area.where((ds_area['lat']>=0.) & (ds_area['lat']<=70.) 
                                 & (ds_area['lon']>=280.) & (ds_area['lon']<=360.))
            dA = dA.compute()
            
            ds_save = area_mean(ds, dA = dA, x='x', y='y')
            
            ds_save = ds_save.get(var_list)
            ds_save = ds_save.compute()
                
            ds_ref = area_mean(ds_ref, dA = dA, x='x', y='y')
            ds_ref = ds_ref.compute()

            # Remove pictr drift
            for var in var_list:
                ds_save[var] = detrend(ds_save[var], ds_ref[var], ['time'])
            
            save_file_path = (save_path + "/" + dir_name + 
                              "/Timeseries/North_Atlantic_thetao_DIC.nc")
            
            ds_save.to_netcdf(save_file_path)
                
            print("Data saved succefully")
        
            
