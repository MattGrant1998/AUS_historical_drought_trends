## Adapted code from Anna Ukkola 

#################################
### IMPORT NECESSARY PACKAGES ###
#################################

from netCDF4 import Dataset,num2date 
import numpy as np
import glob
import sys 
import os
import datetime
import xarray as xr

#################
### Set paths ###
#################

path = "/g/data/w97/amu561/Steven_CABLE_runs/"
# path = "/scratch/w97/mg5624/data/drought_metric/test/"

# lib_path  = str(path + "/scripts/drought_scripts/functions")
lib_path  = str("/home/561/mg5624/RF_project/Anna_code/functions")

# Add lib_path to os directory
sys.path.append(os.path.abspath(lib_path))

#import all droughtmetric calcuation functions
from drought_metrics import *


########################
### DEFINE VARIABLES ###
########################

variable="precip"
### Set drought metric conditions ###
return_all_tsteps=True

### Set percentile for drought threshold ###
perc=15

### Set scale for month aggregation ###
scale=12

### Set to monthly threshold calc ###
monthly=True

### Set additional reference data to false ###
obs_ref = False
obs_var = variable  

### Set historical refernce period ###
# baseline=[1911,2020]
baseline=[1911,2020]

infile=str(path + "/../AGCD_drought_metrics/AGCD_1900_2021/" +
           "AGCD_v1_precip_total_r005_monthly_1900_2021.nc")
# infile = str("/g/data/w97/mg5624/RF_project/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1951_2020.nc")

#################
### Load data ###
#################

    
fh       = Dataset(infile, mode='r')
all_data = fh.variables[variable][:] #[yr_ind]

data     = all_data.data 
#mask     = all_data.mask #AGCD data is not masked
fh_time  = fh.variables["time"]


#Get lon and lat 
try:
    lat = fh.variables['latitude'][:]
    lon = fh.variables['longitude'][:]
except:
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]

#Get dataset dates
fh_dates = num2date(fh_time[:], fh_time.units, calendar=fh_time.calendar)
fh_years = np.array([y.year for y in fh_dates])

miss_val = -999
#data[mask==True] = miss_val

control_ref = data

###################################################
### Create output file name and check if exists ###
###################################################
     
### Define output path ###
out_path = str("/scratch/w97/mg5624/data/drought_metric/test/" + '/drought_metrics_AGCD/' +str(scale) + '-month/')

if not os.path.exists(out_path):    
    os.makedirs(out_path)
##########################################
# ##########################################
# ##########################################
# ##########################################       
#Create output file name
var_name=variable
out_file = str(out_path + "/drought_metrics_AGCD_precip_" + str(fh_years[0]) + "_" + 
               str(fh_years[-1]) + "_baseline_" + str(baseline[0]) + "_" + 
               str(baseline[-1]) + "_scale_" + str(scale) + ".nc")
               
##########################################
##########################################
##########################################
##########################################
##########################################


#############################
### Find baseline indices ###
#############################

#Get dates
ref_years = fh_years
if obs_ref:
    ref_years = obs_years
    

subset = range(np.where(ref_years == baseline[0])[0][0],
                np.where(ref_years == baseline[1])[0][-1] + 1) #Stupid python indexing
                
################################
### Initialise output arrays ###
################################

if return_all_tsteps:
    save_len = len(data)
else:
    save_len = int(len(data)*(perc/100)*2)

duration          = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan
rel_intensity     = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan
rel_intensity_mon = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan

#intensity     = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan
timing        = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan    
#tseries       = np.zeros((save_len, len(lat), len(lon))) + miss_val # * np.nan    

if monthly:
    threshold    = np.zeros((12, len(lat), len(lon))) + miss_val # * np.nan
else:
    threshold    = np.zeros((len(lat), len(lon))) + miss_val # * np.nan

#########################
### Calculate metrics ###
#########################

#Loop through grid cells
for i in range(len(lat)):
            
    for j in range(len(lon)):
    
            #Calculate metrics if cell not missing
            #if any(~mask[:,i,j]):  
            
                #Calculate metrics
            metric = drought_metrics(mod_vec=data[:,i,j], lib_path=lib_path, perc=perc, 
                                    monthly=monthly, obs_vec=control_ref[:,i,j],
                                    return_all_tsteps=return_all_tsteps, scale=scale,
                                    add_metrics=(['rel_intensity', 'threshold', 'rel_intensity_monthly',
                                    'timing']),
                                    subset=subset, miss_val=miss_val)
    
            ### Write metrics to variables ###
            duration[range(np.size(metric['duration'])),i,j]   = metric['duration']  #total drought duration (months)

            rel_intensity[range(np.size(metric['rel_intensity'])),i,j] = metric['rel_intensity'] #average magnitude

            rel_intensity_mon[range(np.size(metric['rel_intensity_monthly'])),i,j] = metric['rel_intensity_monthly'] #average magnitude
        
            #intensity[range(np.size(metric['intensity'])),i,j] = metric['intensity'] #average intensity

            timing[range(np.size(metric['timing'])),i,j]       = metric['timing']    #drought timing (month index)

            #tseries[range(np.size(metric['tseries'])),i,j]       = metric['tseries']    #drought timing (month index)

            if monthly:
                threshold[:,i,j] = metric['threshold'][0:12]    #drought timing (month index)
            else:
                threshold[i,j]   = metric['threshold']


##############################
### Write result to NetCDF ###
##############################
            
# Open a new netCDF file for writing
ncfile = Dataset(out_file,'w', format="NETCDF4_CLASSIC") 

# Create the output data
# Create the x, y and time dimensions
ncfile.createDimension('lat', lat.shape[0])
ncfile.createDimension('lon', lon.shape[0])
ncfile.createDimension('time', save_len)
    
if monthly:
    ncfile.createDimension('month', 12)

# Create dimension variables

longitude = ncfile.createVariable("lon",  'f8', ('lon',))
latitude  = ncfile.createVariable("lat",  'f8', ('lat',))
time      = ncfile.createVariable("time", 'i4', ('time',))

if monthly:
    month = ncfile.createVariable("month", 'i4', ('month',))

#Create data variables
data_dur  = ncfile.createVariable('duration', 'f8',('time','lat','lon'), fill_value=miss_val)
data_mag  = ncfile.createVariable('rel_intensity','f8',('time','lat','lon'), fill_value=miss_val)
data_rel  = ncfile.createVariable('rel_intensity_by_month','f8',('time','lat','lon'), fill_value=miss_val)

#data_int  = ncfile.createVariable('intensity','f8',('time','lat','lon'), fill_value=miss_val)
data_tim  = ncfile.createVariable('timing',   'i4',('time','lat','lon'), fill_value=miss_val)
#data_ts   = ncfile.createVariable('tseries',   'i4',('time','lat','lon'), fill_value=miss_val)

#Create data variable for threshold
if monthly:
    data_thr = ncfile.createVariable('threshold', 'f8',('month','lat','lon'), fill_value=miss_val)
else:
    data_thr = ncfile.createVariable('threshold', 'f8',('lat','lon'), fill_value=miss_val)


#Set variable attributes
longitude.units = 'degrees_east'
latitude.units  = 'degrees_north'
time.units      = 'days since 1960-01-01'

time.calendar   = 'gregorian'

data_dur.long_name = 'drought event duration (no. months)'
data_mag.long_name = 'drought event relative intensity (%)'
data_rel.long_name = 'drought month relative intensity (%)'

#data_int.long_name = 'drought event intensity (mm)'
data_tim.long_name = 'drought event timing (binary drought/non-drought index)'
data_thr.long_name = 'drought threshold (mm)'
#data_ts.long_name  = 'original time series'

if monthly:
    month[:] = range(1,12+1)

# Write data to dimension variables
longitude[:]=lon
latitude[:] =lat

#If saving all time steps
if return_all_tsteps:
    time[:] = fh_time[:]
else:
    time[:] = range(1, save_len+1)
        
if monthly:
    month[:] = range(1,12+1)

#Write data to data variables
data_dur[:,:,:] = duration    
data_mag[:,:,:] = rel_intensity
data_rel[:,:,:] = rel_intensity_mon

#data_int[:,:,:] = intensity
data_tim[:,:,:] = timing
#data_ts[:,:,:]  = tseries

if monthly:    
    data_thr[:,:,:] = threshold
else:
    data_thr[:,:] = threshold

# Close the file
ncfile.close()

