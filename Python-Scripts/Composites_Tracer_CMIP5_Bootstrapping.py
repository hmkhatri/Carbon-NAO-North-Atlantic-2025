"""
This script can be used to compute confidence intervals using boorstrapping method.
The script is set up to compute confidence intervals for tracer budget terms in the North Atlnatic Ocean.
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

ppdir = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

source_id = ['NOAA-GFDL/GFDL-ESM2M/', 'IPSL/IPSL-CM5A-LR/', 'CCCma/CanESM2/', 'MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

tracer_list = ['tos', 'dissic'] # surface fields

num_sample = 1000
cf_lev = 0.95
win_month = 6 # 12-month window averaging for smoother profiles

for exp in experiment_id:
    for model in source_id:

        print("Running bootstrap intervals for model: ", model)

        # get all historical run ensemble members
        dir_list = glob.glob(ppdir + model + "/historical/r*")
        
        for tracer in tracer_list:

            ds1 = []
            # Run loop over ensemble members
            for dir1 in dir_list:
                dir_name = dir1.split('/')[-1].split(',')[0]

                ds_tmp = xr.Dataset()

                d = xr.open_dataset(ppdir + model + "/historical/" + dir_name + 
                                    "/Composites/" + tracer + ".nc")

                for region in ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']:
                    ds_tmp[tracer + region] = d[tracer + region].drop('time')

                ds1.append(ds_tmp)
            ds1 = xr.concat(ds1, dim='r')
            ds1 = ds1.transpose('r', 'time')

            # Compute bootstrapping confidence intervals
            ds_save = xr.Dataset()
            var_list1 = list(ds1.keys()) # get variables in ds1 dataset

            for var1 in var_list1:
            
                dim_list = list(ds1[var1].dims[1:])
                
                data_var = ds1[var1]
                data_var = Moving_Avg(data_var, time = d['time'], time_len = win_month)
                bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                            
                sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)                    
                cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)                    
                cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
                
                ds_save[var1] = data_var.mean('r')
                ds_save[var1 + '_standard_error'] = sde
                ds_save[var1 + '_confidence_lower'] = cfd_low
                ds_save[var1 + '_confidence_upper'] = cfd_up
            
                ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " +
                                                str(cf_lev*100) + "%. ")

            # Save Data
            save_file_path = (ppdir + model + exp + "/all_ensembles/Timeseries/" + tracer + ".nc")

            # Check if the directory exists
            directory = ppdir + model + exp + "/all_ensembles/Timeseries/"
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)

            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)

            print("Data saved succefully for tracer = ", tracer)
        