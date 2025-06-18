"""
This script can be used to area-integrated tracer budget terms (e.g. heat, DIC) using cmip6 simulations. 
Currently, the script is set up for different regions over North Atlantic.

The scripts is set up to save tracer budget terms at different depth levels.

This script makes use of xarray, dask, xgcm libraries for computations and working with netcdf files.
The code can work in serial mode as well as in parallel (see below for details).

With dask, chunksizes are very important. Generally, it is good to have chunksize of 10-100 MB.
For this script, it is recomended to use chunks={'time':1}.
However, the most efficient chunking varies from dataset to dataset. Some manual testing is required to find the most suitable chunking method.

For parallelization, daks-mpi (http://mpi.dask.org/en/latest/) is used to initialize multiple workers on a dask client. This is to ensure that dask is aware of multiple cores assigned through slurm batch submission job.

Instead of dask-mpi, one could also use dask jobqueue (http://jobqueue.dask.org/en/latest/), which is very effective for interactive dask session on jupyter notebook or ipython console.

To run in serial, use "python file.py"
To run in parallel, use "mpirun -np NUM_Cores python file.py"
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
from xgcm import Grid
import xmip.preprocessing as xmip
import glob
import os

import warnings
warnings.filterwarnings("ignore")

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()

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

def Compute_Tracer_Divergence(Field, Velocity, grid, dim, delta = 1., const_multi = 1.):
    
    """Compute velocity times tracer gradient  
    Parameters
    ----------
    Field : xarray DataArray - tracer field
    Velocity : xarray DataArray - velocity along any cartesian direction
    grid : xgcm Grid object
    dim : strig - dimension name
    delta : xarray DataArray - grid spacing along velocity direction
    const_multi : constant - multiplier
    
    Returns
    -------
    Transport : xarray DataArray for advective transport of tracer gradient
    """

    Field_Div = grid.diff(Field, [dim], boundary='extend') / delta # differentiate velocity (extend ensures grandients vanish at boundary)
    Vel_Div_Field = Velocity * Field_Div
    Vel_Div_Field = grid.interp(Vel_Div_Field, [dim], boundary='extend') # interpolate field to tracer grid
    
    Vel_Div_Field = Vel_Div_Field * const_multi
    
    return Vel_Div_Field

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = ['MOHC/UKESM1-0-LL/']
experiment_id = ['historical']

var_list = ['thetao', 'dissic', 'so']

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

RAD_EARTH = 6.387e6

for exp in experiment_id:
    for model in source_id:

        # read grid cell areas and compute grid cell thicknesses on tracer grid
        ds_area = xr.open_mfdataset(cmip_dir + model +  
                                    "piControl/r1i1p1f2/Ofx/areacello/gn/latest/*nc", chunks=-1)
        ds_area = ds_area.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
        
        ds_area['dx'] = ((- ds_area['vertices_longitude'].isel(vertices=slice(0,4,3)).mean('vertices') 
                          + ds_area['vertices_longitude'].isel(vertices=slice(1,3)).mean('vertices')) 
                         * np.cos(ds_area['latitude'] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.))
        
        ds_area['dy'] = ((ds_area['vertices_latitude'].isel(vertices=slice(2,4)).mean('vertices') 
                          - ds_area['vertices_latitude'].isel(vertices=slice(0,2)).mean('vertices')) 
                         * (2 * np.pi * RAD_EARTH / 360.))
        
        ds_area = ds_area.compute()
        
        ds_area = xmip_wrapper(ds_area)

        # get all historical run ensembles 

        dir_list = glob.glob(cmip_dir + model + exp + "/r*")

        for dir1 in dir_list:

            print("Ensemble member running: ", dir1)

            dir_name = dir1.split('/')[-1].split(',')[0]

            # Check ocean carbon data exists

            directory = cmip_dir + model + exp + "/" + dir_name + "/Omon/dissic"
            
            if not os.path.exists(directory): # skip if no data
                continue
        
            for var in var_list:
    
                print("Running for Tracer - ", var)
    
                # Read velocity data 
                var_path = cmip_dir + model + exp +  "/" + dir_name + "/Omon/wo/gn/latest/wo*nc"
                ds_w = xr.open_mfdataset(var_path, chunks={'time':10})
                ds_w = ds_w.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
                ds_w = xmip_wrapper(ds_w)
                ds_w = ds_w.drop(['lat', 'lon', 'lat_bounds', 'lon_bounds'])
                
                var_path = cmip_dir + model + exp +  "/" + dir_name + "/Omon/vo/gn/latest/vo*nc"
                ds_v = xr.open_mfdataset(var_path, chunks={'time':10})
                ds_v = ds_v.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
                ds_v = xmip_wrapper(ds_v)
                ds_v = ds_v.drop(['lat', 'lon', 'lat_bounds', 'lon_bounds'])
                    
                var_path = cmip_dir + model + exp +  "/" + dir_name + "/Omon/uo/gn/latest/uo*nc"
                ds_u = xr.open_mfdataset(var_path, chunks={'time':10})
                ds_u = ds_u.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
                ds_u = xmip_wrapper(ds_u)
                ds_u = ds_u.drop(['lat', 'lon', 'lat_bounds', 'lon_bounds'])
    
                # Read relevant tracer data
                var_path = cmip_dir + model + exp +  "/" + dir_name + "/Omon/" + var + "/gn/latest/" + var + "*nc"
                ds_var = xr.open_mfdataset(var_path, chunks={'time':10})
                ds_var = ds_var.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
                ds_var = xmip_wrapper(ds_var)
                
                ds = xr.merge([ds_var.get([var, 'lat', 'lon']), ds_area.get(['areacello', 'dx', 'dy'])])
    
                # Read relevant tracer air-sea flux data
                if(var == 'thetao'):
                    flux_var = 'hfds'
                elif(var == 'dissic'):
                    flux_var = 'fgco2'
                elif(var == 'so'):
                    flux_var = 'wfo'
                else:
                    print("No flux variable provided")
    
                var_path = (cmip_dir + model + exp +  "/" + dir_name + "/Omon/" + flux_var + 
                            "/gn/latest/" + flux_var + "*nc")
                ds_flx = xr.open_mfdataset(var_path, chunks={'time':10})
                ds_flx = ds_flx.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
                ds_flx = xmip_wrapper(ds_flx)
    
                # combine data - use proper grid indexing (u, v, w are on the same grid)  
                ds = xr.merge([ds, ds_w['wo'].rename({'lev':'lev_w'}), ds_v['vo'].rename({'y':'y_c'}),
                               ds_u['uo'].rename({'x':'x_c'}), 
                               ds_flx[flux_var].drop('time').isel(time=slice(0, len(ds['time'])))])
    
    
                print("Data reading completed")
                
                # Create grid object and get dx, dy on u/v grids for budget calculations
                grid = Grid(ds, coords={'Z': {'center': 'lev_w', 'right': 'lev'},
                                        'Y': {'center': 'y', 'right': 'y_c'},
                                        'X': {'center': 'x', 'right': 'x_c'}}, periodic=[],)
                
                ds['dx_u'] = grid.interp(ds['dx'], 'X', boundary='extend') # interpolate dx to v grid
                ds['dy_v'] = grid.interp(ds['dy'], 'Y', boundary='extend') # interpolate dy to u grid
                
                ds['dz_w'] = (ds_w['lev_bounds'].isel(time=0,bnds=1) -
                              ds_w['lev_bounds'].isel(time=0,bnds=0)).drop('time').rename({'lev':'lev_w'})
                
                # Velocity times trace grandient, u * div(phi)
                ds[var + '_u_div_x'] = Compute_Tracer_Divergence(ds[var], ds.uo, grid = grid, dim = 'X', 
                                                                  delta = ds['dx_u'], const_multi = 1.)
                ds[var + '_u_div_y'] = Compute_Tracer_Divergence(ds[var], ds.vo, grid = grid, dim = 'Y', 
                                                                  delta = ds['dy_v'], const_multi = 1.)
                ds[var + '_u_div_z'] = Compute_Tracer_Divergence(ds[var], ds.wo, grid = grid, dim = 'Z', 
                                                                  delta = ds['dz_w'], const_multi = 1.)

                # Further decompose these terms 
                #use climate-mean tracer concentrations with time-varying velocity
                ds_mean = (ds.get([var, 'uo', 'vo', 'wo'])).mean('time').compute()
                
                ds[var + '_mean_u_div_x'] = Compute_Tracer_Divergence(ds_mean[var], ds.uo, grid = grid, dim = 'X', 
                                                                  delta = ds['dx_u'], const_multi = 1.)
                ds[var + '_mean_u_div_y'] = Compute_Tracer_Divergence(ds_mean[var], ds.vo, grid = grid, dim = 'Y', 
                                                                  delta = ds['dy_v'], const_multi = 1.)
                ds[var + '_mean_u_div_z'] = Compute_Tracer_Divergence(ds_mean[var], ds.wo, grid = grid, dim = 'Z', 
                                                                  delta = ds['dz_w'], const_multi = 1.)

                #use time-varying tracer concentrations with climate-mean velocity
                ds[var + '_u_mean_div_x'] = Compute_Tracer_Divergence(ds[var], ds_mean.uo, grid = grid, dim = 'X', 
                                                                  delta = ds['dx_u'], const_multi = 1.)
                ds[var + '_u_mean_div_y'] = Compute_Tracer_Divergence(ds[var], ds_mean.vo, grid = grid, dim = 'Y', 
                                                                  delta = ds['dy_v'], const_multi = 1.)
                ds[var + '_u_mean_div_z'] = Compute_Tracer_Divergence(ds[var], ds_mean.wo, grid = grid, dim = 'Z', 
                                                                  delta = ds['dz_w'], const_multi = 1.)

        
                #ds = ds.astype(np.float32)
    
                # Perform area-integration
                #cell_area = ds['areacello'] 
                
                # check if having a 3D array for cell area has any effect 
                #(it must not, othere there could be a bug)
                # checked - there is no effect of changing cellarea from 2D to 3D
                cell_area = xr.ones_like(ds[var].isel(time=0)) * ds['areacello']
                cell_area = cell_area * (ds[var].isel(time=0) / ds[var].isel(time=0)).fillna(0.) # remove values for in-land grid cells
                cell_area = cell_area.compute()
                
                save_var_list = [var, var + '_u_div_x', var + '_u_div_y', var + '_u_div_z', flux_var,
                                 var + '_mean_u_div_x', var + '_mean_u_div_y', var + '_mean_u_div_z',
                                 var + '_u_mean_div_x', var + '_u_mean_div_y', var + '_u_mean_div_z']
                
                ds_save = xr.Dataset()
                ds_save['dz'] = (ds_var['lev_bounds'].isel(time=0,bnds=1) 
                                 - ds_var['lev_bounds'].isel(time=0,bnds=0)).drop('time')

                
                # Below with changes in regional boxes (chosen based on extent of temperature and DIC anomalies due to NAO)
                # 1. Subpolar North Atlantic
                dA = cell_area.where((ds['lat']>=48.) & (ds['lat']<=65.) & (ds['lon']>=295.) & (ds['lon']<=340.))
                for save_var in save_var_list:
                    var_area_int = area_sum(ds[save_var], dA = dA, x='x', y='y')
                    ds_save[save_var + '_North_Atlantic_Subpolar'] = var_area_int.compute()#.persist()
                    ds_save[save_var + '_North_Atlantic_Subpolar'].attrs['long_name'] = ("Subpolar North Atlantic " + save_var + 
                                                                                        " (48N-65N, 65W-20W) area-integrated")
                    
                # 2. Subtropical North Atlantic
                dA = cell_area.where((ds['lat']>=25.) & (ds['lat']<=47.) & (ds['lon']>=280.) & (ds['lon']<=340.))
                for save_var in save_var_list:
                    var_area_int = area_sum(ds[save_var], dA = dA, x='x', y='y')
                    ds_save[save_var + '_North_Atlantic_Subtropical'] = var_area_int.compute()#.persist()
                    ds_save[save_var + '_North_Atlantic_Subtropical'].attrs['long_name'] = ("Subtropical North Atlantic " + save_var + 
                                                                                      " (25N-47N, 80W-20W) area-integrated")

                
                
                # Save data 
                # Check if the directory exists
                directory = save_path + model + exp + "/" + dir_name + "/Timeseries"
                if not os.path.exists(directory):
                    # If it doesn't exist, create it
                    os.makedirs(directory)
                
                save_file_path = (save_path + model + exp + "/" + dir_name + 
                                  "/Timeseries/" + var + "_Budget_new_regions_2.nc")
                
                ds_save = ds_save.astype(np.float32).compute()
                ds_save.to_netcdf(save_file_path)
            
                print("Data saved succefully")
            
                ds_save.close()
                ds.close()
                