"""
The script is used for computing time-mean (1850-2014) profiles of temperatures and DIC, and theri meridional transports.
r2i1p1f2 historical simulation is used.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
import os

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

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"
save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

var_list = ['thetao', 'dissic']

RAD_EARTH = 6.387e6

ds = []

for exp in experiment_id:
    for model in source_id:

        # grid information
        ds_area = xr.open_mfdataset(cmip_dir + model + 
                                    "piControl/r1i1p1f2/Ofx/areacello/gn/latest/*nc", chunks=-1)
        mask = xr.open_mfdataset(cmip_dir + model + "/piControl/r1i1p1f2/Ofx/basin/gn/latest/*nc", chunks=-1)

        tmp_dx = ((- mask['vertices_longitude'].isel(vertices=slice(0,4,3)).mean('vertices') 
                   + mask['vertices_longitude'].isel(vertices=slice(1,3)).mean('vertices')) 
                  * np.cos(mask['latitude'] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.))
        
        ds_area['dx'] = tmp_dx.where((mask['basin'] >1.9) & (mask['basin'] < 2.1)) # North Atlantic mask
        
        ds.append(ds_area)
        
        # tracer fields
        for var in var_list:
            d = xr.open_mfdataset(cmip_dir + model + exp +  "/r2i1p1f2/Omon/" + var + 
                                  "/gn/latest/" + var + "*nc", chunks={'lev':1})
            ds.append(d[var])

        ds = xr.merge(ds)
        ds = xmip_wrapper(ds)
        ds = ds.isel(x=slice(180, 310), y=slice(130,300)) # only North Atlantic region

        # Trace Transport fields
        for var in var_list:
            d1 = xr.open_dataset(save_path +  "/r1i1p1f2/Timeseries/" + var + "_Transport_Atlantic.nc")
            
            ds[var + '_Transport'] = (d1[var + '_Transport'] - d1[var + '_Transport_baro']).sum('lev')
            ds[var + '_Transport_overturning'] = d1[var + '_Transport_overturning'].sum('lev')
            ds[var + '_Transport_gyre'] = ds[var + '_Transport'] - ds[var + '_Transport_overturning']

        ds['lat_transport'] = d1['latitude']

        # Compute time-mean and save data
        ds_save = ds.mean('time').compute()
        save_file_path = (save_path + "/r2i1p1f2" + "/Climatology_thetao_DIC.nc")  
        ds_save.to_netcdf(save_file_path)
                
        print("Data saved succefully")


