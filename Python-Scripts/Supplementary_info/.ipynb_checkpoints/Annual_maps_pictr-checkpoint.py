"""
This script can be used to composites of anomalies in upper-100 m temperatures, salinity, DIC and nutrients
in response to a single North Atlantic Oscillation (NAO) in cmip picontrol runs.
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

def annual_mean_data(ds, var_name, num_days, method = 'mean'):
    
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

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = 'MOHC/UKESM1-0-LL/'
experiment_id = 'piControl'

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

# remove 30-year moving mean
num_year = 30
mov_avg = num_year * 12

nao_cut_coef = 2.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (2, 20) # give range of years before / after NAO onset for composites

var_list = ['thetao', 'so', 'dissic', 'mlotst', 'no3', 'po4']

for var in var_list:

    ds1 = xr.open_mfdataset(cmip_dir + source_id + experiment_id + "/r1i1p1f2/Omon/" + 
                            var + "/gn/latest/" + var + "*nc", chunks={'time':1})

    nao = xr.open_dataset(save_path + source_id + experiment_id + "/Timeseries/NAO_SLP.nc", use_cftime=True)
    
    ds1 = xr.merge([ds1, nao.drop('time').isel(time=slice(0, len(ds1['time'])))])
            
    ds1 = xmip_wrapper(ds1)

    if(var == 'mlotst'):
        ds1 = ds1.isel(x=slice(180, 320), y=slice(180, 320)) # only North Atlantic data
    else:
        ds1 = ds1.isel(x=slice(180, 320), y=slice(180, 320), lev = 0)

    ds_basin = xr.open_mfdataset("/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl" + 
                                 "/r1i1p1f2/Ofx/basin/gn/latest/*nc")
    ds_basin = ds_basin.rename({'i':'x','j':'y'})
    ds_basin = ds_basin.isel(x=slice(180, 320), y=slice(180, 320))
    
    ds1 = ds1.compute()
    ocean_var = ds1.get([var])

    # Remove linear drift
    NAO = (ds1['NAO_station'])
    NAO = detrend(NAO, ['time'])
    ocean_var[var] = detrend(ocean_var[var], ['time'])

    # Remove climatology
    NAO_clim = NAO.groupby('time.month').mean('time').compute()
    NAO = NAO.groupby('time.month') - NAO_clim
    
    Var_clim = ocean_var.groupby('time.month').mean('time').compute()
    ocean_var = ocean_var.groupby('time.month') - Var_clim

    # High-pass filter
    if(num_year > 0):
        var_smooth = Moving_Avg(ocean_var[var], time = ocean_var['time'], time_len = mov_avg)
        ocean_var[var] = (ocean_var[var] - var_smooth)
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
                tmp = annual_mean_data(tmp, [var], tmp['time'], method = 'mean')
                
                #tmp = tmp.compute()
                ds_ens_var.append(tmp)

        ds_ens_var = xr.concat(ds_ens_var, dim='r')
        print("Variable: ", var, ", members = ", len(ds_ens_var['r']), ", case: ", cas)

        # Combine NAO+ and NAO- members
        if (cas == 'NAOp'):
            ds_res1_var.append(ds_ens_var)
        elif (cas == 'NAOn'):
            d2 = ds_ens_var
            ds_ens_var[var] = - d2[var]
            ds_res1_var.append(ds_ens_var)

    ds_res_var = xr.concat(ds_res1_var, dim='r')

    ds_save = xr.Dataset()
    ds_save[var + '_mean'] = ds_res_var[var].mean('r')
    ds_save[var + '_std'] = ds_res_var[var].std('r')
    ds_save['num_r'] = len(ds_res_var['r'])

    ds_save = ds_save.assign_coords({'lat': ds_basin['latitude'], 'lon': ds_basin['longitude']})

    ds_save.attrs['description'] = ("Annual anomaly maps at lag years " + 
                                    "-2, -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13-14, 15-16 (year is Oct-Sep)")

    save_file_path = (save_path + source_id + experiment_id + "/Annual_Maps/" + var + ".nc")

    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("Data saved succefully")

    ds_save.close()
    ds1.close()
    

    



