#!/usr/bin/env python
# coding: utf-8

# """
# 
# Last change on Tue Aug 27 2024
# 
# 
# Calculation of calibrations factors between instruments and corresponding errors
# 
# 
# data filter adapted from Herkommer et al. 2024:
# Herkommer, B. and Alberti, C. and Castracane, P. and Chen, J. and Dehn, A. and Dietrich, F. and Deutscher, 
# N. M. and Frey, M. M. and Groß, J. and Gillespie, L. and Hase, F. and Morino, I. and Pak, N. M. and Walker, 
# B. and Wunch, D.:
# Using a portable FTIR spectrometer to evaluate the consistency of Total Carbon Column Observing Network (TCCON) 
# measurements on a global scale: the Collaborative Carbon Column Observing Network (COCCON) travel standard
# Atmos. Meas. Tech., 17, 3467-3494
# https://amt.copernicus.org/articles/17/3467/2024/
# doi     = 10.5194/amt-17-3467-2024
# 
# 
# costfunction adapted from Frey et al. 2015:
# Frey, M., Hase, F., Blumenstock, T., Groß, J., Kiel, M., Mengistu Tsidu, G., Schäfer, K., Sha, M. K., and Orphal, J.: 
# Calibration and instrumental line shape characterization of a set of portable FTIR spectrometers for detecting greenhouse 
# gas emissions,
# Atmos. Meas. Tech., 8, 3047–3057, 
# https://doi.org/10.5194/amt-8-3047-2015.
# 
# 
# with inputs from andreas.luther@tum.de
# 
# 
# @author: lgrosch@iup.physik.uni-bremen.de
# 
# """

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from scipy.optimize import minimize


# # To be changed before running the code:
#     - path to reference data (ref) and data to be calibrated (cal)
#     - instrument properties (XAir and XGas accuracies)
#     - date for data to be evaluated


# # Path to data


# XCO2, XCO and XCH4 data is stored in .csv files in:

X_path = "./calibration_routine/"


# Location of side-by-side measurements

station = "Bremen"


# Data used for reference

ref_start_date = "240508"          # format: YYMMDD

ref_end_date = "240514"            # format: YYMMDD

ref_instr = "HR"


# Read in data used for reference

ref_da = pd.read_csv(os.path.join(
                        X_path, "comb_invparms_" + station + "_" + ref_instr + "_" + ref_start_date + "-" + ref_end_date + ".csv"),
                     sep="\t",
                    )


# Data to be calibrated

cal_start_date = "240508"          # format: YYMMDD

cal_end_date = "240514"            # format: YYMMDD

cal_instr = "SN082"


# Read in data to be calibrated

cal_da = pd.read_csv(os.path.join(
                        X_path, "comb_invparms_" + station + "_" + cal_instr + "_" + cal_start_date + "-" + cal_end_date + ".csv"),
                     sep="\t",
                    )


# # Required functions

# Filter function for outliers and high SZA

def filter_data(input_da, lower_xair, upper_xair):
    
    # Remove values with solar zenith angles larger than 80 degrees
    sza_filtered = input_da[input_da.appSZA <= 80]
    
    # Remove XAir outliers
    xair_filtered = sza_filtered[(sza_filtered["XAIR"] <= upper_xair) & (sza_filtered["XAIR"] >= lower_xair)]
    
    # Remove data outliers
    output_da = xair_filtered[
    (xair_filtered["XCO2"] <= 450) & (xair_filtered["XCO2"] >= 350) & 
    (xair_filtered["XCO"] <= 0.2) & (xair_filtered["XCO"] >= 0.04) & 
    (xair_filtered["XCH4"] <= 1.95) & (xair_filtered["XCH4"] >= 1.6)]

    return output_da


# Data preparation function for bin averaging

def prep_data(input_data):

    # Sort according to time
    input_data.sort_values(by=["UTC"], ignore_index=True)  
    
    
    # Round away microseconds
    input_data["UTC"] = pd.to_datetime(input_data["UTC"]).round("1s") 
    
    
    # Group data to 10 min intevals
    grouped = input_data.groupby(pd.Grouper(key="UTC", axis=0, freq="10T"))
    counts = grouped.size()
    
    
    # Keep only data with at least four values per bin
    filtered_groups = grouped.filter(lambda x: len(x) >= 4)
       

    # Compute the mean of the values within the 10-minute intervals
    averaged_groups = (
        filtered_groups.groupby(pd.Grouper(key="UTC", axis=0, freq="10T"))
        .mean()
        .reset_index()
    )
    
    # Add the 'count' column indicating number of values per bin to the averaged groups for error calculation later
    count_series = counts[averaged_groups.index].reset_index(drop=True)
    averaged_groups['Count'] = count_series.values
    
    
    # Drop NaN columns and rows with all NaN
    output_first = averaged_groups.dropna(axis=1, how='all')
    output_data = output_first.dropna(axis=0, how='any')
    

    return output_data


# Define the cost function

def cost_function(cal_factor, cal_values, ref_values, cal_errors, ref_errors):
    residual = np.power((cal_values/cal_factor - ref_values)/np.sqrt(cal_errors**2+ref_errors**2), 2)
    return np.sum(residual)


# Optimize calibration factor for each species

def find_calibration_factor(cal_day, ref_day, species):
    
    # Initial guess for calibration factor
    initial_cal_factor = 1.0
    
    # Extract data for the given species
    cal_values = cal_day[species].values
    ref_values = ref_day[species].values
    cal_errors = cal_day[species].values
    ref_errors = ref_day[species].values
    
    # Minimize cost function
    result = minimize(cost_function, initial_cal_factor, args=(cal_values, ref_values, ref_errors, cal_errors), method='SLSQP', bounds=[(0, None)])
    
    return result.x[0]  # Optimal calibration factor


# Error calculation

def calculate_weighted_std_dev(cal_values, ref_values, cal_errors, ref_errors, cal_factor):
    # Calculate K = cal / ref for every single bin
    K = cal_values / ref_values
    
    # Compute uncertainties in K using propagation of errors
    sigma_K = K * np.sqrt((cal_errors / cal_values)**2 + (ref_errors / ref_values)**2)
    
    # Calculate the weighted squared deviations between K values and the calibration factor
    weighted_squared_deviations = (K - cal_factor)**2 / sigma_K**2
    
    # Calculate the weighted variance
    weighted_variance = np.sum(weighted_squared_deviations) / np.sum(1 / sigma_K**2)
    
    # Calculate the weighted standard deviation
    weighted_std_dev = np.sqrt(weighted_variance)
    
    return weighted_std_dev


# # Properties of the reference instrument (ref) and the instrument to be calibrated (cal)

# Limits for used XAir values

ref_lower_xair = 0.997
ref_upper_xair = 1.003

cal_lower_xair = 0.998
cal_upper_xair = 1.002


# Standard deviation of single measurement (derived by analysis of time interval with constant XGas abundance)

ref_std_XCO2 = 0.12      #ppm
ref_std_XCO  = 0.00019   #ppb
ref_std_XCH4 = 0.00072   #ppm

cal_std_XCO2 = 0.08      #ppm
cal_std_XCO  = 0.00014   #ppb
cal_std_XCH4 = 0.00047   #ppm


# # Data preparation

# Remove outliers in XAir, XGas and values for high solar zenith angles

ref_filtered = filter_data(ref_da, ref_lower_xair, ref_upper_xair)
cal_filtered = filter_data(cal_da, cal_lower_xair, cal_upper_xair)


# Transform data to ten minute averages

ref = prep_data(ref_filtered)
cal = prep_data(cal_filtered)


# Find matching data

time_matches = set(cal["UTC"]).intersection(ref["UTC"])

ref = ref[ref["UTC"].isin(time_matches)]
cal = cal[cal["UTC"].isin(time_matches)]



# Set new column indexes for calculation of calibration factor

ref.set_index("UTC", inplace=True)
cal.set_index("UTC", inplace=True)


# Calcualte standard error for single bins

ref['Error_XCO2'] = ref_std_XCO2 / np.sqrt(ref['Count'])            
cal['Error_XCO2'] = cal_std_XCO2 / np.sqrt(cal['Count'])
ref['Error_XCO'] = ref_std_XCO / np.sqrt(ref['Count'])          
cal['Error_XCO'] = cal_std_XCO / np.sqrt(cal['Count'])
ref['Error_XCH4'] = ref_std_XCH4 / np.sqrt(ref['Count'])        
cal['Error_XCH4'] = cal_std_XCH4 / np.sqrt(cal['Count'])


# # Calculate day calibration factor

# Extract data for day of interest

date = "2024-05-08"      #yyyy-mm-dd

cal_day = cal.loc[date]
ref_day = ref.loc[date]


# Define species list
species = ["XCO2", "XCO", "XCH4"]

# Calculate calibration factors and errors for each species
cal_factors = {}
sigma_cal_factors = {}

for specie in species:
    # Calculate calibration factor
    cal_factors[specie] = find_calibration_factor(cal_day, ref_day, specie)
    
    # Calculate error for the calibration factor
    sigma_cal_factors[specie] = calculate_weighted_std_dev(
        cal_values=cal_day[specie],
        ref_values=ref_day[specie],
        cal_errors=cal_day[f'Error_{specie}'],
        ref_errors=ref_day[f'Error_{specie}'],
        cal_factor=cal_factors[specie]
    )


# Print daily mean calibration factors for CO2, CO, and CH4

print(f"For the date {date}, the calibration factors are:")
for specie in species:
    cal_factor = round(cal_factors[specie], 6)
    sigma_cal_factor = round(sigma_cal_factors[specie], 6)
    print(f"  - {specie}: {cal_factor} ± {sigma_cal_factor}")

print(f"over {len(cal_day)} bins.")
