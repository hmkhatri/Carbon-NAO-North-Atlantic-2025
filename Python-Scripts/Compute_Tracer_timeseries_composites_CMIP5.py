"""
The script is set up for computing surface tracer times-series using CMIP5 models outputs.
The timeseries is then used to create NAO-based composites to assess the impact of
the NAO on ocean tracer on seasonal-decadal timescales.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os
import numpy as np
import scipy.stats as sc
import xmip.preprocessing as xmip
import sys

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

def area_mean(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-sums
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_mean : timeseris of spatially-averaged dataset
    """
    
    ds_mean = (ds * dA).sum([x, y]) / dA.sum([x, y])
    
    return ds_mean

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
cmip5_dir = "/badc/cmip5/data/cmip5/output1/"
cmip6_dir = "/badc/cmip6/data/CMIP6/CMIP/" 

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

# 3 cmip5 models and 1 cmip6 model. compute area-mean composites in the same code for consistency and simplicity 
#source_id = ['NOAA-GFDL/GFDL-ESM2M/', 'IPSL/IPSL-CM5A-LR/', 'CCCma/CanESM2/', 'MOHC/UKESM1-0-LL/']
source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

# remove 30-year moving mean
num_year = 30
mov_avg = num_year * 12

nao_cut_coef = 2.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (2, 20) # give range of years before / after NAO onset for composites

var_list = ['tos', 'dissic'] # surface fields

for exp in experiment_id:
    for model in source_id:
        
        # get all historical run ensemble members
        dir_list = glob.glob(save_path + model + exp + "/r*")
        
        # Run loop over ensemble members
        for dir1 in dir_list:

            print("Ensemble member running: ", dir1)
            
            dir_name = dir1.split('/')[-1].split(',')[0]
            ds_NAO = xr.open_dataset(save_path + model + exp + "/" + dir_name + 
                                     "/Timeseries/NAO_SLP.nc", 
                                     chunks={'time':-1}, use_cftime=True)

            # Loop over variables
            for var1 in var_list:

                if(var1 == 'tos'):
                    if(model == 'MOHC/UKESM1-0-LL/'):
                        file_path = (cmip6_dir + model + exp + "/" + dir_name + "/Omon/" + 
                                     "thetao" + "/gn/latest/thetao*nc")
                    else:
                        file_path = (cmip5_dir + model + exp + "/mon/ocean/Omon/" + 
                                     dir_name + "/latest/" + var1 + "/" + var1 + "*nc")
                elif(var1 == 'dissic'):
                    if(model == 'MOHC/UKESM1-0-LL/'):
                        file_path = (cmip6_dir + model + exp + "/" + dir_name + "/Omon/" + 
                                     var1 + "/gn/latest/" + var1 + "*nc")
                    else:
                        file_path = (cmip5_dir + model + exp + "/mon/ocnBgchem/Omon/" + 
                                     dir_name + "/latest/" + var1 + "/" + var1 + "*nc")

                d1 = xr.open_mfdataset(file_path, chunks={'time':-1}, use_cftime=True)
                if(model == 'MOHC/UKESM1-0-LL/'):
                    d1 = d1.isel(lev = 0) # only surface data
                    if(var1 == 'tos'):
                        d1[var1] = d1['thetao'] # rename varname

                # read cell_area
                if(model == 'MOHC/UKESM1-0-LL/'):
                    area_path = (cmip6_dir + model + "piControl/r1i1p1f2/Ofx/areacello/gn/latest/")
                else:
                    area_path = (cmip5_dir + model + exp + "/fx/ocean/fx/r0i0p0/latest/areacello/")
                    
                ds_area = xr.open_mfdataset(area_path + "areacello*nc")

                d1 = xr.merge([d1, ds_area['areacello']])

                if(model == 'CCCma/CanESM2/'):
                    d1 = d1.rename({'lon': 'x', 'lat': 'y'})
                elif(model == 'MOHC/UKESM1-0-LL/'):
                    d1 = xmip_wrapper(d1)
                else:
                    d1 = d1.rename({list(d1['lat'].dims)[1]: 'x', list(d1['lat'].dims)[0]: 'y'})
                    d1 = xmip_wrapper(d1)

                ds = xr.merge([d1.get([var1, 'areacello']), 
                               ds_NAO['NAO_station'].drop('time').isel(time=slice(0, len(d1['time'])))]) # get the same length in time

                if(model == 'MOHC/UKESM1-0-LL/'):
                    ds = ds.isel(x = slice(180,310), y = slice(130,300)) # indices range (North Atlantic Ocean)
                else:
                    ds = ds.isel(y=slice(int(len(d1['y'])/2), len(d1['y']))) # only North Atlantic surface data
                
                ds = ds.compute()

                ## Compute area-averaged data
                ocean_var = xr.Dataset()
                cell_area = ds['areacello'] * (ds[var1].isel(time=0) / ds[var1].isel(time=0)).fillna(0.)
                
                # 1. Subpolar North Atlantic 
                if(model == 'CCCma/CanESM2/'):
                    if(var1 == 'tos'): # (50N-65N, 60W-30W)
                        dA = cell_area.where((ds['y']>=50.) & (ds['y']<=65.) & 
                                             (ds['x']>=300.) & (ds['x']<=330.))
                    elif(var1 == 'dissic'): # (50N-60N, 50W-10W)
                        dA = cell_area.where((ds['y']>=50.) & (ds['y']<=60.) & 
                                             (ds['x']>=310.) & (ds['x']<=350.))
                        
                else:
                    if(var1 == 'tos'):
                        dA = cell_area.where((ds['lat']>=50.) & (ds['lat']<=65.) & 
                                             (ds['lon']>=300.) & (ds['lon']<=330.))
                    elif(var1 == 'dissic'):
                        dA = cell_area.where((ds['lat']>=50.) & (ds['lat']<=60.) & 
                                             (ds['lon']>=310.) & (ds['lon']<=350.))

                var_area_mean = area_mean(ds[var1], dA = dA, x='x', y='y')
                ocean_var[var1 + '_North_Atlantic_Subpolar'] = var_area_mean.compute()

                # 2. Sub-Tropical North Atlantic (30N-45N, 60W-25W)
                if(model == 'CCCma/CanESM2/'):
                    dA = cell_area.where((ds['y']>=30.) & (ds['y']<=45.) & 
                                         (ds['x']>=300.) & (ds['x']<=335.))
                else:
                    dA = cell_area.where((ds['lat']>=30.) & (ds['lat']<=45.) & 
                                         (ds['lon']>=300.) & (ds['lon']<=335.))

                var_area_mean = area_mean(ds[var1], dA = dA, x='x', y='y')
                ocean_var[var1 + '_North_Atlantic_Subtropical'] = var_area_mean.compute()

                # Remove linear drift
                NAO = (ds['NAO_station'])
                NAO = detrend(NAO, ['time'])

                for region in ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']:
                    ocean_var[var1 + region] = detrend(ocean_var[var1 + region], ['time'])
            
                # Remove climatology
                NAO_clim = NAO.groupby('time.month').mean('time').compute()
                NAO = NAO.groupby('time.month') - NAO_clim
                
                Var_clim = ocean_var.groupby('time.month').mean('time').compute()
                ocean_var = ocean_var.groupby('time.month') - Var_clim

                # High-pass filter
                if(num_year > 0):
                    for region in ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']:
                        var_smooth = Moving_Avg(ocean_var[var1 + region], time = ocean_var['time'], 
                                                time_len = mov_avg)
                        ocean_var[var1 + region] = (ocean_var[var1 + region] - var_smooth)
                    
                    var_smooth = Moving_Avg(NAO, time = NAO['time'], time_len = mov_avg)
                    NAO = (NAO - var_smooth)

                # Season NAO indices and composite creation
                NAO_season = NAO.isel(time=slice(2,len(NAO.time)-1)).compute() # get rid of first Jan-Feb and last Dec for seasonal avg
                NAO_season = NAO_season.resample(time='QS-DEC').mean('time')
                
                nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
                nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')
        
                ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
                ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)

                ds_res1_var = [] # dataset for composite timeseries
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
                            
                            ds_ens_var.append(tmp.drop('time'))
                            tim = tmp['time']

                    ds_ens_var = xr.concat(ds_ens_var, dim='r')
                    print("Variable: ", var1, ", members = ", len(ds_ens_var['r']), ", case: ", cas)
                    
                    # Combine NAO+ and NAO- members
                    if (cas == 'NAOp'):
                        ds_res1_var.append(ds_ens_var)
                    elif (cas == 'NAOn'):
                        d2 = ds_ens_var.copy()
                        for region in ['_North_Atlantic_Subpolar', '_North_Atlantic_Subtropical']:
                            ds_ens_var[var1 + region] = - d2[var1 + region]
                        ds_res1_var.append(ds_ens_var)

                # Save data 
                ds_res_var = xr.concat(ds_res1_var, dim='r')
                ds_res_var = ds_res_var.assign(time = tim)
                save_file_path = (save_path + model + exp + "/" + dir_name + "/Composites/" + var1 + ".nc")

                # Check if the directory exists
                directory = save_path + model + exp + "/" + dir_name + "/Composites"
                if not os.path.exists(directory):
                    # If it doesn't exist, create it
                    os.makedirs(directory)

                ds_res_var = ds_res_var.astype(np.float32).compute()
                ds_res_var.to_netcdf(save_file_path)

                print("Data saved succefully")