

import xarray as xr
import ast
import numpy as np
import pandas as pd


def preprocess(ds):     
    """This function aligns Sentinel-3 OLCI multiband imagery"""
    
    band_num = ast.literal_eval(ds.bands_rw)
    bands = ['Rw' + str(i) for i in band_num]    
     
    ds_new =  xr.Dataset(
        {'Rrs_400': xr.DataArray(
            data = ds.variables[bands[0]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
        
         'Rrs_412': xr.DataArray(
            data = ds.variables[bands[1]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
        
        'Rrs_443': xr.DataArray(
            data = ds.variables[bands[2]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
         
        'Rrs_490': xr.DataArray(
            data = ds.variables[bands[3]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
         
        'Rrs_510': xr.DataArray(
            data = ds.variables[bands[4]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),

        'Rrs_560': xr.DataArray(
            data = ds.variables[bands[5]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
         
         'Rrs_620': xr.DataArray(
            data = ds.variables[bands[6]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),

        'Rrs_665': xr.DataArray(
            data = ds.variables[bands[7]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),

        'Rrs_674': xr.DataArray(
            data = ds.variables[bands[8]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'}),
         
        'Rrs_682': xr.DataArray(
            data = ds.variables[bands[9]]/np.pi, dims=['height', 'width'], coords={'Date':pd.to_datetime(ds.start_time.split(' ')[0]), 'lat':ds.variables['latitude'][:], 'lon':ds.variables['longitude'][:]}, attrs  = {'units':'sr^-1'})})
         
    return ds_new




def cdom_calc(rrs_443, rrs_490, rrs_510, rrs_560, rrs_665):
    """This function applies an optimized multilinear regression for estimating CDOM
    to Sentinel-3 OLCI multi-band satellite imagery"""
    
    
    c1 = -0.27036093; c2 = 3.44358941; c3 = -3.8377943; c4 = -0.06177415; c5 = 1.15006086; c6 = 4.227966776705502
    ag300 = np.exp(c1 * np.log(rrs_443) + c2 * np.log(rrs_490) +
                      c3 * np.log(rrs_510) + c4 * np.log(rrs_560) +
                      c5 * np.log(rrs_665) + c6)
    return ag300



def s275_calc(rrs_443, rrs_490, rrs_510, rrs_560, rrs_665):
    """This function applies an optimized multilinear regression for estimating spectral slope (265-295)
    to Sentinel-3 OLCI multi-band satellite imagery"""
    
    d1 = -0.17042979; d2 = 1.43449794; d3 = -1.68159208; d4 = 0.69000059; d5 = -0.28499721; d6 = -4.233559628382057
    s275 = np.exp(d1 * np.log(rrs_443) + d2 * np.log(rrs_490) +
                      d3 * np.log(rrs_510) + d4 * np.log(rrs_560) +
                      d5 * np.log(rrs_665) + d6)
    return s275


def oc4_calc(rrs_443, rrs_490, rrs_510,rrs_560):
    """This function applies the OC4 chlorphyll a algorithm to Sentinel-3 OLCI bands"""
    a0 = 0.42540; a1 = -3.21679; a2 = 2.86907; a3 = -00.62628; a4 = -1.09333
    
    X = np.log10( np.nanmax([rrs_443, rrs_490, rrs_510], axis=0) / rrs_560)
    chl = 10 ** (a0 + a1 * X + a2 * X **2 + a3 * X**3 + a4 * X ** 4)
    return chl