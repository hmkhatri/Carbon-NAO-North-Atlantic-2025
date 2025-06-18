"""
This script can be used to compute confidence intervals using boorstrapping method.
The script is set up to compute confidence intervals for tracer budget terms in the North Atlantic Ocean.
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

def Data_time_tendency(data, time_val, month_window):
    """Computes time-derivative using a timeseries
    Parameters
    ----------
    data : xarray DataArray for data
    time_val : xarray DataArray for time values
    month_window : int number of months for computing time derivative
    
    Returns
    -------
    data_dt : xarray DataArray for d(data) / dt
    """
    
    for i in range(0, month_window):
        if(i == 0):
            day_count = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) * 0.5 # day count
        elif(i < win_month-1):
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - 
                                                                              month_window + i)).drop('time') 
        else:
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - 
                                                                              month_window + i)).drop('time') * 0.5

    print(day_count.shape, len(data['time']), month_window)
    
    data_dt = ((-data.isel(time = slice(0, len(data['time']) - month_window)).drop('time') + 
                data.isel(time = slice(month_window, len(data['time']))).drop('time')) / (24. * 3600. * day_count.drop('time')))

    data_dt = data_dt.assign_coords(time=time_val.isel(time=slice(int(np.floor(month_window/2)), 
                                                                  len(data['time']) - month_window + int(np.floor(month_window/2)))))
    
    return data_dt

### ------ Main calculations ------------------

ppdir = ("/gws/nopw/j04/unicorns/carbon_add_redis" + 
         "/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/")
dir_list = glob.glob(ppdir + "r*")

num_sample = 1000
cf_lev = 0.95
win_month = 12 # 12-month window averaging for smoother profiles

fac = [31./1.e3] # multiplier to have nutrient budget in kg/s
depth_levels = [100., 1000.] #, 10000.] 
var_list = ['po4']

case = ['NAOp', 'NAOn']

for depth in depth_levels:
    print('Depth Level: ', depth) 
    depth_lev = [0., depth] # for budget analysis
    k = 0
    
    clim_MLD = xr.open_dataset(ppdir + "/all_ensembles/Climatology/MLD_thetao_dissic.nc")
    
    for var1 in var_list:
        
        ind_check = 0
        ds1 = []
        
        for dir1 in dir_list:
            dir_name = dir1.split('/')[-1].split(',')[0]
            
            # combine nao+ and nao- data
            for cas in case:
                # New runs in newer regions with decomposing into time-varying and time-mean contributions
                d1 = xr.open_dataset(ppdir + dir_name + "/Composites/" + cas + 
                                     "_Composite_" + var1 + "_Budget_new_regions_2.nc")
                
                d1 = d1.drop(['P_south', 'P_north', 'NAO_station'])
                
                if(cas == 'NAOp'):
                    if(ind_check == 0):
                        ds1.append(d1.drop(['dz']))
                        ind_check = 1
                    else:
                        ds1.append(d1.drop(['dz','time']))
                
                else:
                    d2 = d1.drop(['dz']).copy()
                    for var in list(d2.keys()):
                        d1[var] = - d2[var]
                        
                    ds1.append(d1.drop(['dz','time']))
                    
        ds1 = xr.concat(ds1, dim='r')
        ds1 = ds1.transpose('r', 'time', 'lev')
        
        save_var_list = [var1, var1 + '_u_div_x', var1 + '_u_div_y', var1 + '_u_div_z',
                         var1 + '_mean_u_div_x', var1 + '_mean_u_div_y', var1 + '_mean_u_div_z',
                         var1 + '_u_mean_div_x', var1 + '_u_mean_div_y', var1 + '_u_mean_div_z']
    
        region_list = ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']
    
        for region in region_list:
            for var in save_var_list:

                tmp = (ds1[var + region] * d1['dz']).sel(lev=slice(depth_lev[0],
                                                                   depth_lev[1])).sum('lev') * fac[k]
                    
                if(var == var1):
                    ds1['dt_' + var + region] = Data_time_tendency(tmp, ds1['time'], win_month)
                    
                ds1[var + region] = Moving_Avg(tmp, time = ds1['time'], time_len = win_month)

            if(var1 == 'po4'):               
                particulate_flux = - ds1['epp100' + region] * fac[k]
                ds1['particulate_flux' + region] = Moving_Avg(particulate_flux, time = ds1['time'],
                                                              time_len = win_month)
            
            ds1[var1 + '_hor_div' + region] = (- ds1[var1 + '_u_div_x' + region] 
                                               - ds1[var1 + '_u_div_y' + region]) 
            ds1[var1 + '_ver_div' + region] = (ds1[var1 + '_u_div_z' + region])
            
            ds1[var1 + '_mean_hor_div' + region] = (- ds1[var1 + '_mean_u_div_x' + region] 
                                                    - ds1[var1 + '_mean_u_div_y' + region]) 
            ds1[var1 + '_mean_ver_div' + region] = (ds1[var1 + '_mean_u_div_z' + region])
            
            ds1[var1 + '_u_mean_hor_div' + region] = (- ds1[var1 + '_u_mean_div_x' + region] 
                                                      - ds1[var1 + '_u_mean_div_y' + region]) 
            ds1[var1 + '_u_mean_ver_div' + region] = (ds1[var1 + '_u_mean_div_z' + region])
            
            ds1[var1 + '_res' + region] = (ds1['dt_' + var1 + region] - ds1[var1 + '_ver_div' + region] 
                                           - ds1[var1 + '_hor_div' + region]) # residual term
            
            if(var1 == 'po4'):
                ds1 = ds1.drop(['o2sat' + region, 'o2' + region, 'epp100' + region])
                ds1[var1 + '_res_bio' + region] = (ds1[var1 + '_res' + region] 
                                                   - ds1['particulate_flux' + region]) # residual term

        # Bootstrapping analysis
        ds_save = xr.Dataset()
        
        var_list1 = list(ds1.keys()) # get variables in ds dataset
        
        for var in var_list1:
            
            dim_list = list(ds1[var].dims[1:])
            
            data_var = ds1[var]
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                        
            sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)                    
            cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)                    
            cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
            
            ds_save[var] = ds1[var].mean('r')
            ds_save[var + '_standard_error'] = sde
            ds_save[var + '_confidence_lower'] = cfd_low
            ds_save[var + '_confidence_upper'] = cfd_up
            
        ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " +
                                        str(cf_lev*100) + "%. ")

        # New runs in newer regions with decomposing into time-varying and time-mean contributions
        save_file_path = (ppdir + "all_ensembles/Timeseries/" + var1 + "_Budget_new_regions_2_depth_" 
                          + str(int(depth_lev[1])) + ".nc")
        
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
        
        print("Data saved succefully for = ", var1)
        
        k = k + 1
        
        ds1.close()
        ds_save.close()
    
    











