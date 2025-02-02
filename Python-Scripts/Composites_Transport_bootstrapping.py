"""
This script can be used to compute confidence intervals using boorstrapping method.
The script is set up to compute confidence intervals for meridional overturning and tracer transport diagnostics.
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

num_sample = 1000
cf_lev = 0.95

#var_list = ['thetao', 'dissic', 'so']
var_list = ['po4']

case = ['NAOp', 'NAOn']

for var1 in var_list:
    
    ind_check = 0
    ds1 = []
    
    for dir1 in dir_list:
        dir_name = dir1.split('/')[-1].split(',')[0]
        
        # combine nao+ and nao- data
        for cas in case:
            
            d1 = xr.open_dataset(ppdir + dir_name + "/Composites/" + cas + 
                                 "_Composite_" + var1 + "_Transport_Atlantic.nc")
            d1 = d1.drop(['P_south', 'P_north', 'NAO_station'])
            
            if(cas == 'NAOp'):
                if(ind_check == 0):
                    ds1.append(d1.drop(['latitude']))
                    ind_check = 1
                else:
                    ds1.append(d1.drop(['latitude','time']))
            
            else:
                d2 = d1.drop(['latitude']).copy()
                for var in list(d2.keys()):
                    d1[var] = - d2[var]
                    
                ds1.append(d1.drop(['latitude','time']))
                
    ds1 = xr.concat(ds1, dim='r')
    ds1 = ds1.transpose('r', 'y', 'time', 'lev')

    print("Data reading complete for variable: ", var1)

    # compute mean response function and errorbars with bootstrapping
    ds1 = Moving_Avg(ds1, time = ds1['time'], time_len = 6) # 6-month running mean for smoother profiles
    ds1 = ds1.isel(time=slice(36,240)) # get shorter timeslice for efficiency (whole dataset not required)
    
    ds_save = xr.Dataset()

    lat_ind = [15, 19, 25, 31] # corresponding close to 10N, 25N, 45N, 60N     
    for var in list(ds1.keys()):

        dim_list = list(ds1[var].dims[2:])
        data_mean = []; lat_val = [] 
        sde_var1 = []; cfd_up_var1 = []; cfd_low_var1 = []
        
        for lat in range(0, len(ds1['y'])):
            
            data_var = ds1[var].isel(y=lat).compute()
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)

            sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
            sde_var1.append(sde)
                        
            cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
            cfd_up_var1.append(cfd_up)
                        
            cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
            cfd_low_var1.append(cfd_low)
            
            data_mean.append(data_var.mean('r'))
            lat_val.append(d1['latitude'].isel(y=lat))
        
        ds_save[var] = xr.concat(data_mean, dim='lat')
        ds_save['latitude'] = xr.concat(lat_val, dim='lat')
        ds_save[var + '_standard_error'] = xr.concat(sde_var1, dim='lat') 
        ds_save[var + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='lat') 
        ds_save[var + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='lat')

    ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval:" + 
                                    "Meridional tracer transport response to a single NAO+ event." + 
                                    " Confidence interval is " + str(cf_lev*100) + "%. ")
    
    save_file_path = (ppdir + "all_ensembles/Timeseries/" + var1 + "_Transport_Atlantic.nc")

    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("Data saved succefully")


            