"""
This script can be used to create composites of budgets for DIC, tempearture, salt timeseries based on extreme NAO indices using cmip historical runs.
The following steps are taken.  
1. Long-term linear trend is removed from picontrol runs.
2. Variations at timescales longer than 30 years are removed.
3. Composites are created based on NAO+ and NAO- events.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import glob
import os

import warnings
warnings.filterwarnings("ignore")

# ----------- Functions -------------
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


### ------ Main calculations ------------------

dir_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/MOHC/UKESM1-0-LL/historical/"

nao_cut_coef = 2.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (5, 30) # give range of years before / after NAO onset for composites

# remove 30-year moving mean
num_year = 30
mov_avg = num_year * 12

#var_list = ['thetao', 'dissic', 'so'] 
var_list = ['po4']

dir_list = glob.glob(dir_path + "r*")

for dir1 in dir_list:

    print("Ensemble member running: ", dir1)

    dir_name = dir1.split('/')[-1].split(',')[0]
    
    ds_NAO = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/NAO_SLP.nc", use_cftime=True)
    ds_MLD = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/mlotst.nc", use_cftime=True)
    
    for var1 in var_list:
        # Earlier runs without decomposing into time-varying and time-mean contributions
        #d1 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/" + var1 + "_Budget.nc", use_cftime=True)
        
        # New runs with decomposing into time-varying and time-mean contributions
        #d1 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/" + var1 + "_Budget_2.nc", use_cftime=True)
        
        #ds = xr.merge([d1.drop('dz'), ds_MLD.drop('time'),
        #               ds_NAO.drop('time').isel(time=slice(0, len(d1['time'])))]) # get the same length in time

        # New runs in newer regions with decomposing into time-varying and time-mean contributions
        d1 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/" + var1 + "_Budget_new_regions_2.nc", use_cftime=True)
        
        ds = xr.merge([d1.drop('dz'), ds_MLD.drop('time'),
                       ds_NAO.drop('time').isel(time=slice(0, len(d1['time'])))]) # get the same length in time
        
        if(var1 == 'dissic'):
            d1 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/o2.nc", use_cftime=True)
            d2 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/o2sat.nc", use_cftime=True)
            d3 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/expc.nc", use_cftime=True)
            d4 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/epc100.nc", use_cftime=True)
            d5 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/epcalc100.nc", use_cftime=True)
            d6 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/phyc.nc", use_cftime=True)
            ds = xr.merge([ds, d1.drop('dz'), d2.drop('dz'), d3, d4, d5, d6]) # add oxygen data for preformed and regenerated calculations
        elif(var1 == 'po4'):
            d1 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/o2.nc", use_cftime=True)
            d2 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/o2sat.nc", use_cftime=True)
            d3 = xr.open_dataset(dir_path +  "/" + dir_name + "/Timeseries/epp100.nc", use_cftime=True)
            ds = xr.merge([ds, d1.drop('dz'), d2.drop('dz'), d3])
    
        # Remove linear drift
        for var in list(ds.keys()):
            ds[var] = detrend(ds[var], ['time'])
    
        # Remove climatology
        ds_clim = ds.groupby('time.month').mean('time')
        ds = ds.groupby('time.month') - ds_clim
        
        # High-pass filter
        if(num_year > 0):
            for var in list(ds.keys()):
                var_smooth = Moving_Avg(ds[var], time = ds['time'], time_len = mov_avg)
                ds[var] = (ds[var] - var_smooth)
    
        print("Data reading complete for var: ", var1)
        
        # Compute NAO indices
        NAO = (ds['NAO_station'])
        NAO = NAO.isel(time=slice(2,len(NAO.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        NAO_season = NAO.resample(time='QS-DEC').mean('time')
    
        nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
        nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')
    
        # create composites
        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)
    
        for cas in case:
        
            ds_ens = []
    
            if (cas == 'NAOp'):
                count_NAO = ind_NAOp
            elif (cas == 'NAOn'):
                count_NAO = ind_NAOn
            else:
                print("Choose a valid case")
    
            # composites
            for year in range(year_str + int(num_year/2), len(nao_DJF) - year_end*2 - int(num_year/2)):
            
                if(count_NAO.isel(time=year) == 1):
            
                    year_val = nao_DJF['time.year'][year]
    
                    tmp = ds.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)
    
                    ds_ens.append(tmp.drop('time'))
    
                    tim = tmp['time']
    
            ds_ens = xr.concat(ds_ens, dim='r')
            ds_ens = ds_ens.assign(time = tim)
            ds_ens['dz'] = d1['dz']
            
            ds_ens['mlotst_North_Atlantic_Subpolar_mean'] = ds_MLD['mlotst_North_Atlantic_Subpolar'].mean('time')
            ds_ens['mlotst_North_Atlantic_Subtropical_mean'] = ds_MLD['mlotst_North_Atlantic_Subtropical'].mean('time')
            
            # Save data 
            # Check if the directory exists
            directory = dir_path + "/" + dir_name + "/Composites"
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)

            # Earlier runs without decomposing into time-varying and time-mean contributions
            #ds_ens.to_netcdf(dir_path + "/" + dir_name + "/Composites/" + cas + "_Composite_" + var1 + "_Budget.nc")

            # New runs with decomposing into time-varying and time-mean contributions
            #ds_ens.to_netcdf(dir_path + "/" + dir_name + "/Composites/" + cas + "_Composite_" + var1 + "_Budget_2.nc")

            # New runs in newer regions with decomposing into time-varying and time-mean contributions
            ds_ens.to_netcdf(dir_path + "/" + dir_name + "/Composites/" + cas + "_Composite_" + var1 + "_Budget_new_regions_2.nc")
            
            print("Composites data saved successfully for var: ", var1)


