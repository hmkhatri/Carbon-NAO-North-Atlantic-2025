"""
This script can be used to composites of anomlaies in surface temperatures, salinity, DIC and nutrients
in response to a single North Atlantic Oscillation (NAO) in cmip historical runs.
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
    ds = xmip.correct_lon(ds)

    return ds

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

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

def annaul_mean_data(ds, var_name, num_days, method = 'mean'):
    
    """Compute annual mean of data for bootstrapping
    Means are computed for year = -1, 0, 1, 2, 3-4, 5-6
    Parameters
    ----------
    ds : xarray Dataset for data variables
    var_name : list of avariables for computing annual means
    num_days : Number of days in months
    method : Options - compute 'mean', 'integrate', 'difference' over time 
    
    Returns
    -------
    ds_annual : Dataset containting annual means
    """
    
    ds_annual = xr.Dataset()
    
    for var1 in var_name:
        
        data_var1 = []
        
        ind_correct = 0
        for i in range(0,11): # mean for years = -2, -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13-14, 15-16

            if (i<=2):
                days = num_days.dt.days_in_month.isel(time = slice(12*i + 9, 12*i + 12 + 9)) # Oct-Sep annual mean
                data_var = ds[var1].isel(time = slice(12*i + 9, 12*i + 12 + 9))
            else:
                days = num_days.dt.days_in_month.isel(time = slice(12*(i + ind_correct) + 9, 12*(i + ind_correct + 1) + 9 + 12))
                data_var = ds[var1].isel(time = slice(12*(i + ind_correct) + 9, 12*(i + ind_correct + 1) + 9 + 12))
                ind_correct = ind_correct + 1

            if(method == 'mean'):
                data_var = ((data_var * days).sum('time')/ days.sum('time'))
            elif(method == 'integrate'):
                data_var = ((data_var * days).sum('time') * 3600. * 24.)
            elif(method == 'difference'):
                data_var = (data_var.isel(time=-1) - data_var.isel(time=0))
            else:
                print("Method is not valid")
            
            data_var1.append(data_var)
            
        ds_annual[var1] = xr.concat(data_var1, dim='year')
    
    #ds_annual = ds_annual.chunk({'year':-1})
        
    return ds_annual

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/"

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

# remove 30-year moving mean
num_year = 30
mov_avg = num_year * 12

nao_cut_coef = 2.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (2, 20) # give range of years before / after NAO onset for composites

#var_list = ['thetao', 'dissic', 'so', 'o2', 'no3', 'po4', 'o2sat', 'talk', 'chl']
#var_list = ['mlotst', 'zos', wo', 'epc100']

var_list = ['po4', 'chl'] # for anomalies averaged in upper 100-m

dir_list = glob.glob(save_path + "r*")

for dir1 in dir_list:

    print("Ensemble member running: ", dir1)

    dir_name = dir1.split('/')[-1].split(',')[0]
    
    ds_NAO = xr.open_dataset(save_path + "/" + dir_name + "/Timeseries/NAO_SLP.nc", 
                             chunks={'time':-1}, use_cftime=True)
        
    for var1 in var_list:

        # Check var1 data exists
        directory = cmip_dir + "/" + dir_name + "/Omon/" + var1
        if not os.path.exists(directory): # skip if no data
            continue
        
        d1 = xr.open_mfdataset(cmip_dir +  "/" + dir_name + "/Omon/" + 
                               var1 + "/gn/latest/" + var1 + "*nc", 
                               chunks={'time':-1}, use_cftime=True)
        d1 = xmip_wrapper(d1)

        #d1 = d1.isel(x=slice(180, 320), y=slice(180, 320), lev=0) # only North Atlantic surface data
        #d1 = d1.isel(x=slice(180, 320), y=slice(180, 320))
        #d1 = d1.isel(x=slice(180, 320), y=slice(180, 320), lev=24) # for vertical velocities at 100m depth

        d1 = d1.isel(x=slice(180, 320), y=slice(180, 320), lev=slice(0,24)) # upper 100 m
        
        ds = xr.merge([d1.get([var1, 'lat', 'lon']), 
                       ds_NAO['NAO_station'].drop('time').isel(time=slice(0, len(d1['time'])))]) # get the same length in time

        # ------ Use only if averaging over 100 m -------- #
        dz = (d1['lev_bounds'].isel(time=0,bnds=1) - d1['lev_bounds'].isel(time=0,bnds=0)).drop('time')
        dz = dz * (ds[var1].isel(time=0) / ds[var1].isel(time=0)).fillna(0.) # remove values for in-land grid cells

        ds[var1] = (ds[var1] * dz).sum('lev') / dz.sum('lev')
        # ------ Use only if averaging over 100 m -------- #
        
        ds = ds.compute()
    
        # Remove linear drift
        NAO = (ds['NAO_station'])
        NAO = detrend(NAO, ['time'])
        
        ocean_var = ds.get([var1])
        ocean_var[var1] = detrend(ocean_var[var1], ['time'])
    
        # Remove climatology
        NAO_clim = NAO.groupby('time.month').mean('time').compute()
        NAO = NAO.groupby('time.month') - NAO_clim
        
        Var_clim = ocean_var.groupby('time.month').mean('time').compute()
        ocean_var = ocean_var.groupby('time.month') - Var_clim
    
        # High-pass filter
        if(num_year > 0):
            var_smooth = Moving_Avg(ocean_var[var1], time = ocean_var['time'], time_len = mov_avg)
            ocean_var[var1] = (ocean_var[var1] - var_smooth)
            var_smooth = Moving_Avg(NAO, time = NAO['time'], time_len = mov_avg)
            NAO = (NAO - var_smooth)
            
        # Season NAO indices and composite creation
        NAO_season = NAO.isel(time=slice(2,len(NAO.time)-1)).compute() # get rid of first Jan-Feb and last Dec for seasonal avg
        NAO_season = NAO_season.resample(time='QS-DEC').mean('time')

        nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
        nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')

        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)

        ds_res1_var = [] # dataset for annual maps
    
        for cas in case:
            
            ds_ens_var = []
            
            if (cas == 'NAOp'):
                count_NAO = ind_NAOp
            elif (cas == 'NAOn'):
                count_NAO = ind_NAOn
            else:
                print("Choose a valid case")
    
            for year in range(year_str + int(num_year/2), len(nao_DJF) - year_end - int(num_year/2)):
                
                if(count_NAO.isel(time=year) == 1):
                
                    year_val = nao_DJF['time.year'][year]
    
                    tmp = ocean_var
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)
    
                    # compute annual means
                    tmp = annaul_mean_data(tmp, [var1], tmp['time'], method = 'mean')
                    
                    ds_ens_var.append(tmp)
                    
            ds_ens_var = xr.concat(ds_ens_var, dim='r')
            print("Variable: ", var1, ", members = ", len(ds_ens_var['r']), ", case: ", cas)
            
            # Combine NAO+ and NAO- members
            if (cas == 'NAOp'):
                ds_res1_var.append(ds_ens_var)
            elif (cas == 'NAOn'):
                d2 = ds_ens_var
                ds_ens_var[var1] = - d2[var1]
                ds_res1_var.append(ds_ens_var)
                
            ds_res_var = xr.concat(ds_res1_var, dim='r')
            ds_res_var['lon'] = ds['lon']
            ds_res_var['lat'] = ds['lat']

            # Save data 
            # Check if the directory exists
            directory = save_path + "/" + dir_name + "/Annual_Maps"
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
                
            save_file_path = (save_path + "/" + dir_name + "/Annual_Maps/" + var1 + "_100.nc")

            ds_res_var = ds_res_var.astype(np.float32).compute()
            ds_res_var.to_netcdf(save_file_path)
    
            print("Data saved succefully")
            
    