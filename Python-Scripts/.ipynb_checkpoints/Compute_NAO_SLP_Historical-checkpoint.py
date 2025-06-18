"""
This script uses mean sea level pressure for computing NAO index using cmip6 simulations.
NAO indices are computes duing two methods
1. Differeces betwen mean sea level pressure over Icealand and Azores regions
2. EOF-PCA analysis over the North Atlantic sector and PC1 corresponds to NAO.

See for details,
https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-station-based
https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-pc-based
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
import os

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)

    return ds

def compute_grid_areas(ds, x='X', y='Y', RAD_EARTH = 6.387e6):

    """Compute grid-cell areas
    Parameters
    ----------
    ds : xarray Dataset for data variables
    
    Returns
    -------
    Cell_area : grid cell areas
    """

    ds = ds.copy()

    dx = np.mean(ds[x].diff(x)) * np.cos(ds[y] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.)
    dy = np.mean(ds[y].diff(y)) * (2 * np.pi * RAD_EARTH / 360.)

    Cell_area = dx * dy

    Cell_area, tmp = xr.broadcast(Cell_area, ds['psl'].isel(time=0))

    return Cell_area


### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

var = 'psl'

RAD_EARTH = 6.387e6

for exp in experiment_id:

    for model in source_id:
        
        # get all historical run ensembles 

        dir_list = glob.glob(cmip_dir + model + exp + "/r*")

        for dir1 in dir_list:

            dir_name = dir1.split('/')[-1].split(',')[0]

            # Check ocean carbon data exists

            directory = cmip_dir + model + exp + "/" + dir_name + "/Omon/dissic"
            
            if not os.path.exists(directory): # skip if no data
                continue

            # Read sea-lvel pressure data for NAO computations
            ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/" + dir_name + "/Amon/" + var + 
                                    "/g*/latest/" + var + "*.nc", chunks={'time':1}) 
            
            ds1 = xmip_wrapper(ds1)
        
            print("Data reading complete for Ensemple: ", dir_name)
        
            # --------- Compure sea-level pressures and station-based NAO --------- 
            dA = compute_grid_areas(ds1, x='x', y='y', RAD_EARTH = RAD_EARTH) # grid cell areas for area-integration
        
            P_south = ((ds1[var].sel(y = slice(36., 40.), x = slice(332., 340.)) * 
                        dA.sel(y = slice(36., 40.), x = slice(332., 340.))).sum(['x','y']) / 
                       dA.sel(y = slice(36., 40.), x = slice(332., 340.)).sum(['x','y']))
        
            P_north = ((ds1[var].sel(y = slice(63., 70.), x = slice(335., 344.)) * 
                        dA.sel(y = slice(63., 70.), x = slice(335., 344.))).sum(['x','y']) / 
                       dA.sel(y = slice(63., 70.), x = slice(335., 344.)).sum(['x','y']))
            
            ds_save = xr.Dataset()
            
            ds_save['P_south'] = P_south
            ds_save['P_south'].attrs['units'] = "Pa"
            ds_save['P_south'].attrs['long_name'] = "Mean Sea Level Pressure over Azores Region"
            
            ds_save['P_north'] = P_north
            ds_save['P_north'].attrs['units'] = "Pa"
            ds_save['P_north'].attrs['long_name'] = "Mean Sea Level Pressure over Iceland Region"
    
            ds_save['NAO_station'] = (ds_save['P_south'] - ds_save['P_north']) # NAO based on station-pressure differences
            ds_save['NAO_station'].attrs['units'] = "hPa"
            ds_save['NAO_station'].attrs['long_name'] = "Station-based NAO Index"
            
            # ---------  Compute EOF-based NAO indices ---------
            """
            psl_anom = ds1[var].rename({'x':'lon', 'y':'lat'})
            psl_anom = psl_anom.sel(lat=slice(20.,80.)) # latitude range 20N-80N
            psl_anom = psl_anom.where((psl_anom['lon'] <= 40.) & (psl_anom['lon'] >= 270.)) #longitude range 90W-40E

            model = xe.models.EOF(n_modes=10, use_coslat=True)
            model.fit(psl_anom, dim='time')
            #model.solve()
            
            ds_save['expvar'] = model.explained_variance_ratio()
            ds_save['expvar'].attrs['long_name'] = "EOF-PCA: Explained variance"
            
            ds_save['eofs'] = model.components()
            ds_save['eofs'].attrs['long_name'] = "EOF-PCA: EOF modes"
            
            ds_save['pcs'] = model.scores()
            ds_save['pcs'].attrs['long_name'] = "EOF-PCA: Principal Components"

            """
            # Save data 
            # Check if the directory exists
            directory = save_path + model + exp + "/" + dir_name + "/Timeseries"
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
            
            save_file_path = (save_path + model + exp + "/" + dir_name + "/Timeseries/NAO_SLP.nc")
            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)
        
            print("Data saved succefully")
    
            ds1.close()
            
