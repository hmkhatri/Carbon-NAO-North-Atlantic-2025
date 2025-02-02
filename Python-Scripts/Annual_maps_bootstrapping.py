"""
This script can be used to compute confidence intervals using boorstrapping method.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os
import numpy as np
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------
def data_bootstrap(data, cf_lev = 0.95, num_sample = 1000):
    
    """Compute bootstrap confidence intervals and standard error for data along axis =0
    Parameters
    ----------
    data : xarray DataArray for data
    stat : statisctic required for bootstrapping function, e.g. np.mean, np.std
    cf_lev : confidence level
    num_sample : Number of bootstrap samples to generate
    Returns
    -------
    bootstrap_ci : object contains float or ndarray of
        bootstrap_ci.confidence_interval : confidence intervals
        bootstrap_ci.standard_error : standard error
    """
    
    data = (data,)
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci


### ------ Main calculations ------------------

ppdir = ("/gws/nopw/j04/unicorns/carbon_add_redis" + 
         "/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/")
dir_list = glob.glob(ppdir + "r*")

#var_list = ['thetao', 'dissic', 'so', 'o2', 'po4', 'o2sat', 'talk', 'no3', 'chl', 'mlotst', 'zos', 'wo']

#var_list = ['epc100']
var_list =['po4','chl'] 

num_sample = 1000
cf_lev = 0.95

ds_basin = xr.open_mfdataset("/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl" + 
                             "/r1i1p1f2/Ofx/basin/gn/latest/*nc")
ds_basin = ds_basin.rename({'i':'x','j':'y'})
ds_basin = ds_basin.isel(x=slice(180, 320), y=slice(180, 320))

for var in var_list:

    ds_map = []

    for dir1 in dir_list:
        dir_name = dir1.split('/')[-1].split(',')[0]

        # Check var data exists
        #directory = ppdir + "/" + dir_name + "/Annual_Maps/" + var + ".nc"
        #if not os.path.exists(directory): # skip if no data
        #    continue

        # maps at single depth level
        #d1 = xr.open_dataset(ppdir + "/" + dir_name + "/Annual_Maps/" + var + ".nc", chunks={'year':1})

        # upper 100 m mean maps
        d1 = xr.open_dataset(ppdir + "/" + dir_name + "/Annual_Maps/" + var + "_100.nc", chunks={'year':1})
        
        ds_map.append(d1)
        
    ds_map = xr.concat(ds_map, dim='r')
    ds_map = ds_map.drop(['lon','lat'])
        
    ds_map = ds_map.assign_coords({'lat': ds_basin['latitude'], 'lon': ds_basin['longitude']})

    # bootstrapping
    ds_save = xr.Dataset()
    dim_list = list(ds_map[var].dims[2:])
        
    sde_var1 = []; cfd_up_var1 = []; cfd_low_var1 = []
    
    for yr in range(0, len(ds_map['year'])):
            
        data_var = ds_map[var].isel(year=yr).compute()
        bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                    
        sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
        sde_var1.append(sde)
                    
        cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
        cfd_up_var1.append(cfd_up)
                    
        cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
        cfd_low_var1.append(cfd_low)
            
    ds_save[var] = ds_map[var].mean('r')
    ds_save[var + '_standard_error'] = xr.concat(sde_var1, dim='year') 
    ds_save[var + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='year') 
    ds_save[var + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='year')
    
    ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " 
                                    + str(cf_lev*100) + "%. " + 
                                    "Spatial maps at lag years -2, -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13-14, 15-16 (year is Oct-Sep)")
    
    #save_file_path = (ppdir + "all_ensembles/Annual_maps/" + var + ".nc")
    save_file_path = (ppdir + "all_ensembles/Annual_maps/" + var + "_100.nc") # upper 100 m mean maps
    
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
        
    print("Data saved succefully for = ", var)
        




