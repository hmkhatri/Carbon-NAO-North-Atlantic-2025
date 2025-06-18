"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes overturning circulation and meridional tracer/nutrient transport at depth levels.
The script uses UKESM-0-LL dataset.

1. Read v and tracer (temperature, salt, carbon, nutrients) data on z-levels.
2. Compute overturning by integrating transport in the zonal direction as well as in the direction of increasing depth.
3. Compute Meridional heat, freshwater, carbon transport at z-levels.
4. Finally, data is saved in netcdf format.

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
import numpy as np
import xarray as xr
import cf_xarray
import gsw as gsw
from xgcm import Grid
import xmip.preprocessing as xmip
import gc
from tornado import gen
import os
import glob

import warnings
warnings.filterwarnings('ignore')

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

def select_subset(dataset):

    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude',
                            'time_bnds']) # drop variables

    return dataset

def Transport_z(ds_vel, grid, thickness = 1., assign_name='transport'):

    """Compute volume transport per unit horizontal length
    Parameters
    ----------
    ds_vel : xarray DataArray - velocity data
    thickness : xarray DataArray - vertical grid thickness
    grid : xgcm Grid object
    assign_name : name for the volume transport

    Returns
    -------
    transport : xarray DataArray for volume transport
    """
    
    transport =  ds_vel * thickness # velocity x vertical grid spacing
    transport = transport.fillna(0.).rename(assign_name)

    return transport

def Compute_Tracer_transport(Field, Velocity, grid, dim = None, const_multi = 1.):

    """Compute transport of field along velocity component
    Parameters
    ----------
    Field : xarray DataArray - tracer field
    Velocity : xarray DataArray - velocity along meridional (or zonal for zonal transport) direction
    grid : xgcm Grid object
    dim : strig - dimension name
    const_multi : constant - multiplier

    Returns
    -------
    Transport : xarray DataArray for tracer transport
    """

    if(dim != None):
        Field_vel = grid.interp(Field, [dim], boundary='extend') # interpolate tracer field to velocity grid
    else:
        Field_vel = Field

    Transport = Field_vel * Velocity * const_multi

    return Transport

def Compute_barotropic(ds_vel, Field, grid, dim = None, dz = 1., dx = 1., dim_v = 'Z', dim_x = 'X'):
    
    """Compute zonal and depth mean velocity and tracer field
    Parameters
    ----------
    ds_vel : xarray DataArray - velocity data
    Field : xarray DataArray/Datset - tracer field data
    grid : xgcm Grid object
    dim : dimension for interpolation to have tracel and velocity on the same grid
    dz : xarray DataArray - grid cell vertical thickness
    dx : xarray DataArray - Zonal grid spacing
    dim_v : Vertical dimension name
    dim_x : Zonal dimension name

    Returns
    -------
    vel_baro : xarray DataArray for zonal-depth-mean velocity
    Field_baro : xarray DataArray for zonal-depth-mean tracer
    """
    
    if(dim != None): # interpolate Field to velocity data grid
        
        Field_baro = xr.Dataset()
        for var in list(Field.keys()):
            Field_baro[var] = grid.interp(Field[var], [dim], boundary='extend')
    else:
        Field_baro = Field
    
    vel_baro = ((ds_vel * dz * dx).sum(dim=[dim_v, dim_x])
                / (dz * dx).sum(dim=[dim_v, dim_x]))
    
    Field_baro = ((Field_baro * dz * dx).sum(dim=[dim_v, dim_x])
                  / (dz * dx).sum(dim=[dim_v, dim_x]))
    
    return vel_baro, Field_baro

def Compute_Overturning(Transport, dx = 1., dim_v = 'Z', dim_x = 'X', long_name = None):

    """Compute Overturning circulation using meridional velocity x thickness data
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    dx : xarray DataArray - Zonal grid spacing
    dim_v : Vertical dimension name
    dim_x : Zonal dimension name
    long_name : string - long name for the variable

    Returns
    -------
    overturning : xarray DataArray - overturning circulation
    """

    overturning = (Transport * dx).sum(dim=dim_x).cumsum(dim=dim_v)
    overturning.attrs['units'] = "m^3/s"
    overturning.attrs['long_name'] = long_name

    return overturning

def Meridional_Tracer_Transport_Overturning(Transport, Field, dx = 1., dz = 1., dim_x = 'X', dimen = None, long_name = None):

    """Compute Meriidonal tracer transport corresponding to meridional overturning circulation
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    Field : xarray DataArray - Tracer Field * vertical thickness
    dx : xarray DataArray - Zonal grid spacing
    dz : xarray DataArray - Layer thicknesses
    dim_x : Zonal dimension name
    dimen : string - dimentional units for output
    long_name : string - long name for output

    Returns
    -------
    Meridional_Tracer_Transport : xarray DataArray - meridional tracer transport due to overturning circulation
    """

    Transport_Meridional = (Transport * dx).sum(dim=dim_x) # first get net meridional volume transport
    Meridional_Tracer = (Field * dx).sum(dim=dim_x) / ((dz * dx).sum(dim=dim_x) + 1.e-10) # zonal mean of tracer content
    # dz here is f(x,y,z), which is zero for non-wet points. It is to ensure that only wet grid points are considered
    # + 1.e-10 is to avoid 0./0. situations

    Meridional_Tracer_Transport = Transport_Meridional * Meridional_Tracer

    Meridional_Tracer_Transport.attrs['units'] = dimen
    Meridional_Tracer_Transport.attrs['long_name'] = long_name

    return Meridional_Tracer_Transport

def Compute_Meridional_Transport_Components(Volume_Transport, Tracer_Transport, Tracer, v_baro, Tracer_baro, 
                                            dx = 1., dz = 1., dim_x = 'X', const_multi = 1., dimen = None, Tracer_name = 'Tracer'):
    

    """Compute components of meridional tracer transport as a function of vertical coordinate.
    see derivations and technical details in Khatri et al. (2022), GRL, https://doi.org/10.1029/2022GL101480

    Parameters
    ----------
    Volume_Transport : xarray DataArray - meridional velocity x layer thickness
    Tracer_Transport : xarray DataArray - tracer x meridional velocity x layer thickness
    Tracer : xarray DataArray - tracer concentration
    v_baro : xarray DataArray - zonal and depth mean meridional veloctiy
    Tracer_baro : xarray DataArray - zonal and depth mean tracer value
    dx : xarray DataArray - Zonal grid spacing
    dz : xarray DataArray - Layer thicknesses
    dim_x : Zonal dimension name
    const_multi : constant multiplier (e.g. to convert temperatures to heat content)
    dimen : string - dimentional units for output
    Tracer_name : string : tracer name for assigning names for different vars
    
    Returns
    -------
    ds_out : xarray Dataset - meridional tracer transport components
    """

    ds_out = xr.Dataset()

    ds_out[Tracer_name + '_Transport'] = (Tracer_Transport * dx).sum(dim=dim_x) * const_multi
    ds_out[Tracer_name + '_Transport'].attrs['units'] = dimen
    ds_out[Tracer_name + '_Transport'].attrs['long_name'] = "Meridional " + Tracer_name + " transport"

    ds_out[Tracer_name + '_Transport_baro'] = (dz * v_baro * Tracer_baro * const_multi * dx).sum(dim=dim_x)
    ds_out[Tracer_name + '_Transport_baro'].attrs['units'] = dimen
    ds_out[Tracer_name + '_Transport_baro'].attrs['long_name'] = "Meridional " + Tracer_name + " transport - Barotropic v and Barotropic " + Tracer_name

    with xr.set_options(keep_attrs=True):
        tracer_content_z = (Tracer - Tracer_baro) * dz * const_multi
        transport = Volume_Transport - dz * v_baro
        ds_out[Tracer_name + '_Transport_overturning'] = Meridional_Tracer_Transport_Overturning(transport, tracer_content_z, dx = dx, 
                                                                                                 dz = dz, dim_x = dim_x, dimen = dimen,
                                                                                                 long_name = "Meridional " + Tracer_name + 
                                                            " transport due to overturning circulation - Baroclinic v and Baroclinic " + Tracer_name)

        #tracer_content_z = (Tracer - Tracer_baro) * dz * const_multi # already computed above, no need to recompute
        transport1 = dz * v_baro
        ds_out[Tracer_name + '_Transport_baro_v'] = Meridional_Tracer_Transport_Overturning(transport1, tracer_content_z, dx = dx, 
                                                                                            dz = dz, dim_x = dim_x, dimen = dimen,
                                                                                            long_name = "Meridional " + Tracer_name + 
                                                            " transport due to overturning circulation - Barotropic v and Baroclinic " + Tracer_name)
        
        tracer_content_z = Tracer_baro * dz * const_multi
        #transport = Volume_Transport - dz * v_baro # already computed above, no need to recompute
        ds_out[Tracer_name + '_Transport_baro_' + Tracer_name] = Meridional_Tracer_Transport_Overturning(transport, tracer_content_z, 
                                                                                                         dx = dx, dz = dz, dim_x = dim_x, 
                                                                dimen = dimen, long_name = "Meridional " + Tracer_name + 
                                                            " transport due to overturning circulation - Baroclinic v and Barotropic " + Tracer_name)

    return ds_out


### ------------- Main computations ------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"

save_path = "/gws/nopw/j04/unicorns/carbon_add_redis/Fast_SLow_NAO/"

# read mask and meridional velocity data and compute zonal grid spacing
mask = xr.open_mfdataset(cmip_dir + "/piControl/r1i1p1f2/Ofx/basin/gn/latest/*nc")
mask = mask.isel(i=slice(180, 310), j=slice(130,300)) # only North Atlantic region

RAD_EARTH = 6.387e6

tmp_dx = ((- mask['vertices_longitude'].isel(vertices=slice(0,4,3)).mean('vertices') 
           + mask['vertices_longitude'].isel(vertices=slice(1,3)).mean('vertices')) 
          * np.cos(mask['latitude'] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.))

mask['dx'] = tmp_dx.where((mask['basin'] >1.9) & (mask['basin'] < 2.1)) # North Atlantic mask

mask = xmip_wrapper(mask)

# Variable list for computing meridional transports

# ------ use the code below for heat, salt and DIC tranports ------
#rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015
#var_list = ['thetao', 'dissic', 'so']
#S_ref = 35. # Reference salinity in psu

#cons_multiplier = [rho_cp, 1., 1./S_ref] # constants to convert temperatures to heat transport, salt to volume transport
#dimen_list = ['Joules/s','mol/s', 'm^3/s']
# --------------------------

# --- for po4 tranports ------
var_list = ['po4']
cons_multiplier = [1.]
dimen_list = ['mol/s']
# --------------------------
              
# get all historical run ensembles 
dir_list = glob.glob(cmip_dir + 'historical' + "/r*")

for dir1 in dir_list:

    print("Ensemble member running: ", dir1)

    dir_name = dir1.split('/')[-1].split(',')[0]

    # Check ocean carbon data exists
    directory = cmip_dir + 'historical' + "/" + dir_name + "/Omon/dissic"
            
    if not os.path.exists(directory): # skip if no data
        continue

    k = 0
    for var in var_list:
    
        print("Running for Tracer - ", var)
    
        # Read velocity data 
        var_path = cmip_dir + 'historical' +  "/" + dir_name + "/Omon/vo/gn/latest/vo*nc"
        ds_v = xr.open_mfdataset(var_path, chunks={'time':10})
        ds_v = ds_v.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
        ds_v = xmip_wrapper(ds_v)
        ds_v = ds_v.drop(['lat', 'lon', 'lat_bounds', 'lon_bounds'])

        # Read relevant tracer data
        var_path = cmip_dir + 'historical' +  "/" + dir_name + "/Omon/" + var + "/gn/latest/" + var + "*nc"
        ds_var = xr.open_mfdataset(var_path, chunks={'time':10})
        ds_var = ds_var.isel(i = slice(180,310), j = slice(130,300)) # indices range (North Atlantic Ocean)
        ds_var = xmip_wrapper(ds_var)

        # combine data - use proper grid indexing (tracer and v are on different grid) 
        ds = xr.merge([ds_var.get([var, 'lat', 'lon']), mask.get(['dx']),
                      ds_v['vo'].rename({'y':'y_c'})])
        
        ds['dz'] = (ds_var['lev_bounds'].isel(time=0,bnds=1) -
                    ds_var['lev_bounds'].isel(time=0,bnds=0)).drop('time')
        
        cell_dz = xr.ones_like(ds[var].isel(time=0)) * ds['dz']
        cell_dz = cell_dz * (ds[var].isel(time=0) / ds[var].isel(time=0)).fillna(0.) # remove values for in-land grid cells

        level_outer_data = (cf_xarray.bounds_to_vertices(ds_var['lev_bounds'].isel(time=0).chunk({'lev':-1}),'bnds').load().data)
        ds = ds.assign_coords({'level_outer': level_outer_data})
        
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                'Y': {'center': 'y', 'right': 'y_c'}}, periodic=[],)
        ds['vo_t'] = grid.interp(ds['vo'], 'Y', boundary='fill') # interpolate velocity to tracer cells
        
        # --------------------------------------- #
        # Meridional transports are computed by integrating the relevant diagnostics in the zonal direction.
        # Many model outputs are on a irregular grid, so integration along 'x' is not always along a constant latitude, 
        # especially near polar regions.
        # Ideally, one may regrid the data to regular lat-lon grid before performing the below computations.
        # However, for simplicity, we work with data on model native grid. The final results do not change significantly, 
        # which has been observed in many works.
    
        # Here, we compute tracer transpor as a function of vertical coordinate and tracer transport is decomposed into 5 components.
        # see details in Khatri et al. (2022), GRL, https://doi.org/10.1029/2022GL101480
        # --------------------------------------- #
        
        # compute meridional volume and tracer transport on z-levels
        ds_transport = xr.Dataset()
        ds_transport['vo_t'] = Transport_z(ds['vo_t'], grid, thickness = cell_dz, assign_name='Meridional_Transport')
        ds_transport[var] = Compute_Tracer_transport(ds[var], ds_transport['vo_t'], grid = grid,
                                                     dim = None, const_multi = 1.)
        
        # compute overturning and meridional transport
        [v_baro, tracer_baro] = Compute_barotropic(ds['vo_t'], ds.get([var]), grid, dim = None, dz = cell_dz,
                                                       dx = ds['dx'], dim_v='lev', dim_x = 'x')
        
        ds_save = xr.Dataset()
        ds_save['latitude'] = ds['lat'].where(~np.isnan(ds['dx'])).mean('x').compute()
    
        with xr.set_options(keep_attrs=True):
            ds_save['Overturning_z'] = Compute_Overturning(ds_transport['vo_t'], dx = ds['dx'], dim_v = 'lev',
                                                           dim_x = 'x', long_name="Overturning circulation vs depth")
                
            Meridional_Transport_baro = cell_dz * v_baro
            ds_save['Overturning_z_barotropic'] = Compute_Overturning(Meridional_Transport_baro, dx = ds['dx'], dim_v = 'lev', dim_x = 'x', 
                                                                      long_name="Overturning circulation vs depth - Barotropic Component")
    
        with xr.set_options(keep_attrs=True):
            ds_out = Compute_Meridional_Transport_Components(ds_transport['vo_t'], ds_transport[var], ds[var], 
                                                             v_baro, tracer_baro[var], 
                                                             dx = ds['dx'], dz = cell_dz, dim_x = 'x', const_multi = cons_multiplier[k], 
                                                             dimen = dimen_list[k], Tracer_name = var)
        ds_save = xr.merge([ds_save, ds_out])
        
        k = k + 1

        # Save data 
        # Check if the directory exists
        directory = save_path + 'MOHC/UKESM1-0-LL/historical' + "/" + dir_name + "/Timeseries"
        if not os.path.exists(directory):
            # If it doesn't exist, create it
            os.makedirs(directory)
            
        save_file_path = (save_path + 'MOHC/UKESM1-0-LL/historical' + "/" + dir_name + 
                            "/Timeseries/" + var + "_Transport_Atlantic.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
            
        print("Data saved succefully")
            
        ds_save.close()
        ds.close()
        
        

