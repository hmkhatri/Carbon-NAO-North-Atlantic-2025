"""
This scripts computes the volume of wet ocean points at each depth level in specified regions.
Currently, the script is set up to compute cell volumeas a function of depth in three regions.
1. Supolar North Atlantic (45N-60N, 50W-20W)
2. Mid-lat North Atlantic (25N-45N, 60W-30W)
2. Subtropical North Atlantic (10N-25N, 50W-20W)
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob

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

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['piControl']

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

var = 'dissic' # to indetify non-nan grid points

for exp in experiment_id:
    
    for model in source_id:
        
        ds = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f2/Omon/" + var + "/gn/latest/" + var + "*nc", chunks={'time':1})
    
        area = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f2/Ofx/areacello/gn/latest/*nc")
    
        ds = xr.merge([ds, area['areacello']])
            
        ds = xmip_wrapper(ds)

        ds = ds.isel(x = slice(180,310), y = slice(130,300)) # indices range (North Atlantic Ocean)

        # Perform area-integration
        dz = (ds['lev_bounds'].diff('bnds')).isel(bnds=0, time=0).drop('bnds')
        DIC = ds[var] # DIC is dummay var_name
    
        cell_volume = ds['areacello'] * dz
        
        cell_volume = cell_volume.where((DIC.isel(time=0) > -1.e6) & 
                                            (DIC.isel(time=0) < 1.e6)).compute()

        ds_save = xr.Dataset()

        ds_save['dz'] = dz
        
        # 1. Subpolar North Atlantic
        dA = cell_volume.where((ds['lat']>=45.) & (ds['lat']<=60.) & (ds['lon']>=310.) & (ds['lon']<=340.)).compute()
        dV = area_sum(1., dA = dA, x='x', y='y')
        ds_save['Volume_North_Atlantic_Subpolar'] = dV
        ds_save['Volume_North_Atlantic_Subpolar'].attrs['units'] = 'm^3'

        # 2. Mid-lat North Atlantic DIC
        dA = cell_volume.where((ds['lat']>=25.) & (ds['lat']<=45.) & (ds['lon']>=300.) & (ds['lon']<=330.)).compute()
        dV = area_sum(1., dA = dA, x='x', y='y')

        ds_save['Volume_North_Atlantic_Midlat'] = dV
        ds_save['Volume_North_Atlantic_Midlat'].attrs['units'] = 'm^3'

        # 3. Sub-Tropical North Atlantic
        dA = cell_volume.where((ds['lat']>=10.) & (ds['lat']<=25.) & (ds['lon']>=310.) & (ds['lon']<=340.)).compute()
        dV = area_sum(1., dA = dA, x='x', y='y')

        ds_save['Volume_North_Atlantic_Subtropical'] = dV
        ds_save['Volume_North_Atlantic_Subtropical'].attrs['units'] = 'm^3'

        save_file_path = (save_path + model + "historical/all_ensembles/Cell_volume.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
        
        print("Data saved succefully")

        ds_save.close()
        ds.close()

