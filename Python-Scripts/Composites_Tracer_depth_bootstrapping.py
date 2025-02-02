"""
This script can be used to compute confidence intervals using bootstrapping method.
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


### ------ Main calculations ------------------

ppdir = ("/gws/nopw/j04/unicorns/carbon_add_redis" + 
         "/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/")
dir_list = glob.glob(ppdir + "r*")

#var_list = ['thetao', 'dissic', 'so']
var_list = ['dissic']
#region_list = ['_North_Atlantic_Subpolar', '_North_Atlantic_Midlat', '_North_Atlantic_Subtropical']
region_list = ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']

num_sample = 1000
cf_lev = 0.95

for var in var_list:

    # Combine composites from all historical runs
    ds = []
    ds_NAO = []
    
    for dir1 in dir_list:
        dir_name = dir1.split('/')[-1].split(',')[0]
    
        ds1 = xr.Dataset()

        d1 = xr.open_dataset(ppdir + "/" + dir_name + 
                             "/Composites/NAOp_Composite_" + var + "_Budget_new_regions_2.nc")
        d2 = xr.open_dataset(ppdir + "/" + dir_name + 
                             "/Composites/NAOn_Composite_" + var + "_Budget_new_regions_2.nc")
            
        for region in region_list:
            ds1[var + region] = xr.concat([d1[var + region], -d2[var + region].drop('time')], dim='r')
            if(var == 'dissic'):
                ds1['expc' + region] = xr.concat([d1['expc' + region], -d2['expc' + region].drop('time')], dim='r')
                
            # add Mixed-layer depth profiles
            ds1['mlotst' + region] = xr.concat([d1['mlotst' + region], -d2['mlotst' + region].drop('time')], dim='r')
            ds1['mlotst' + region] = ds1['mlotst' + region] + d1['mlotst' + region + '_mean']
    
        ds.append(ds1.drop('time'))
        
        ds2 = xr.concat([d1['NAO_station'], -d2['NAO_station'].drop('time')], dim='r')
        ds_NAO.append(ds2.drop('time'))
        
    ds = xr.concat(ds, dim='r')
    ds_NAO = xr.concat(ds_NAO, dim='r')

    ds = Moving_Avg(ds, time = [1.], time_len = 6) # running mean with 6-month window (smoother profiles)
    
    ds = xr.merge([ds, ds_NAO])
    ds = ds.isel(time=slice(36,300)) # get shorter timeslice for efficiency (whole dataset not required)
    
    # Bootstrapping analysis
    ds_save = xr.Dataset()

    var_list1 = list(ds.keys()) # get variables in ds dataset

    for var1 in var_list1:
        
        dim_list = list(ds[var1].dims[1:])
        
        data_var = ds[var1]
        bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                    
        sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)                    
        cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)                    
        cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
        
        ds_save[var1] = ds[var1].mean('r')
        ds_save[var1 + '_standard_error'] = sde
        ds_save[var1 + '_confidence_lower'] = cfd_low
        ds_save[var1 + '_confidence_upper'] = cfd_up
        
    ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " +
                                    str(cf_lev*100) + "%. ")
    
    save_file_path = (ppdir + "all_ensembles/Timeseries/Depth_time_" + var + ".nc")
    
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
        
    print("Data saved succefully for = ", var)
    


