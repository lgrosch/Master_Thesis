#!/usr/bin/env python
# coding: utf-8

"""

Last change on Tue Aug 27 2024

Calculation of slant column density (molecules/cm²) of the simulated plume

with inputs from Dr. Alexandros Panagiotis Poulidis, University of Bremen

@author: lgrosch@iup.physik.uni-bremen.de

"""

import numpy as np
import netCDF4 as nc
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Load the NetCDF file
file_path = './grid_conc_20240514010000(4).nc'
dataset = nc.Dataset(file_path)

# Extract the dimensions needed
time = dataset.variables['time'][:]
longitude = dataset.variables['longitude'][:]
latitude = dataset.variables['latitude'][:]
heights = dataset.variables['height'][:]
spec001_mr = dataset.variables['spec001_mr'][:]


current_time = start_time + timedelta(seconds=int(dataset.variables['time'][53]))


# Load the solar zenith and azimuth angles from the EM27 CSV file
angles_df = pd.read_csv("./comb_invparms_Stahlwerk_hb_1_SN082_240514-240514.csv", sep="\t",)  # Assumes columns: 'UTC', 'appSZA', 'azimuth'

# Convert 'UTC' column to datetime format
angles_df['UTC'] = pd.to_datetime(angles_df['UTC'])

# Define the start time of the simulation
start_time = datetime(2024, 5, 14, 1, 0, 0)  # 2024-05-14 01:00:00


# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['UTC', 'emission (ng/cm^-2)', 'zenith', 'azimuth'])
#steps_df = pd.DataFrame(columns=['UTC', 'step emission (ng/cm^-2)', 'alt', 'lon', 'lat'])


lat0 = 53.156   # Station latitude
lon0 = 8.629    # Station longitude
alt0 = 0        # Station altitude (ground level)


# Function that finds the indices of the grid that the given point lies within.

def find_grid_indices(lat, lon, alt, lat_grid, lon_grid, alt_grid):
    
    lat_idx = np.abs(lat_grid - lat).argmin()               # choose latitude grid point closest to given one
    lon_idx = np.abs(lon_grid - lon).argmin()               # choose longitude grid point closest to given one
    alt_idx = np.searchsorted(alt_grid, alt, side='right')  # choose next higher altitude level
    return lat_idx, lon_idx, alt_idx


# Iterate through each time index
for time_index in range(spec001_mr.shape[2]):
    
    # Calculate the current UTC time based on time_index
    current_time = start_time + timedelta(seconds=int(dataset.variables['time'][time_index]))
    
    # Find the closest UTC time in the solar angles data
    closest_time = angles_df.iloc[(angles_df['UTC'] - current_time).abs().argmin()]
    
    # Extract the corresponding solar angles
    solar_zenith_angle = closest_time['appSZA']
    solar_azimuth_angle = closest_time['azimuth']
    
    # Convert angles to radians
    zenith_rad = np.deg2rad(solar_zenith_angle)
    azimuth_rad = np.deg2rad(solar_azimuth_angle)
    
    # Calculate maximal model column latitude
    min_latitude = latitude.min()

    # Define vector components for that column
    lat_dy = abs(lat0 - min_latitude)
    meter_dy = lat_dy * 111111.0

    meter_dx = np.tan(azimuth_rad) * meter_dy
    lon_dx = meter_dx / (111111.0 * np.cos(np.deg2rad(lat0)))

    meter_dz = np.sqrt(meter_dx**2+meter_dy**2) * np.tan((np.pi/2)-zenith_rad)
    
    # Direction vector in meter for path length calculation
    degree_vector = np.array([lon_dx, lat_dy, meter_dz])
    meter_vector = np.array([meter_dx, meter_dy, meter_dz])
    column_length = np.sqrt(meter_dx**2+meter_dy**2+meter_dz**2)
    
    # Select the correct time slice and reduce dimensions
    spec001_mr_slice = spec001_mr[0, 0, time_index, :, :, :]  # Shape: (77, 30, 50)

    # Steps along the slant path
    total_steps = 2000 

    # Initialize
    path_length = 0
    total_concentration = 0
    path_concentration_sum = 0
    lat = lat0
    lon = lon0
    alt = alt0

    # Traverse the slant column
    while alt < 2000:
        lat_idx, lon_idx, alt_idx = find_grid_indices(lat, lon, alt, latitude, longitude, heights)

        # Calculate concentration at the current position
        conc = spec001_mr_slice[alt_idx, lat_idx, lon_idx]

        # Calculate the path length within this grid box (approximation)
        path_length = column_length / total_steps

        # Sum the path-concentration product (ng/m^2)
        path_concentration_sum += conc * path_length
                     
        # Update the position
        lon -= degree_vector[0] / total_steps
        lat -= degree_vector[1] / total_steps
        alt += meter_vector[2] / total_steps

    # Convert the result from m^-2 to cm^-2
    total_column_concentration = path_concentration_sum / 10000

    # Store the result in the DataFrame
    results_df = results_df.append({'UTC': current_time,
                                    'emission (ng/cm^-2)': total_column_concentration,
                                    'zenith': solar_zenith_angle,
                                    'azimuth': solar_azimuth_angle}, 
                                   ignore_index=True)

# Save the results to a CSV file
results_df.to_csv('./slant_column_concentration_SW2.csv', sep=';', decimal=',', index=False)


#show and save selected time interval
short=results_df.iloc[48:67]
print(short)
short.to_csv('./slant_column_concentration_SW1.csv', sep=';', decimal=',', index=False)

# Make the plot

plt.figure(figsize=(10, 6))
plt.plot(short['UTC'], short['emission (ng/cm^-2)'], marker='o', linestyle='-', color='b')
plt.xlabel('UTC')
plt.ylabel('Total Column Concentration [ng/cm²]')
plt.title('Total Column Concentration for Slant Column')
plt.grid(True)

# Save the plot to a file
plt.savefig('./slant_column_SW1.png', bbox_inches='tight', dpi=300)  # Adjust dpi for quality

plt.show()

