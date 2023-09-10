import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.fft import fft, ifft

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 2023

@author: magnu
"""

"""
This function takes the Fourier transformed data and makes a cooresponding
frequency axis based on the sampling rate that has to be pre defined as "Hz"

It then plots the spectrum with a grid, the title and labels
"""
def PlotFFT(spectrum, title):
    n = np.arange(len(spectrum))  # Length of the Fourier transform
    T = len(spectrum)/Hz  # sampling period
    frequencies = n/T

    # Plot the result
    plt.figure()
    plt.plot(frequencies[:len(spectrum)//2+1],
             np.abs(spectrum[:len(spectrum)//2+1]))
    plt.grid(which='both')
    plt.minorticks_on()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.show()


Hz = 500
# Problem 1
# ****************************************************************************
# Open file dialog to choose file
root = tk.Tk()
file_path = filedialog.askopenfilename()
root.destroy()

# Load data into np array
data = np.genfromtxt(file_path, delimiter=';', skip_header=1)

# Separate data
measured_data = data[:, 0]
simulateded_data = data[:, 1]
diff_data = simulateded_data-measured_data
print("Data loaded successfully")

# Create time axsis
T = np.linspace(0, m.floor(len(measured_data)/Hz), len(measured_data))

# Measured and simulated
plt.figure()
plt.plot(T, measured_data, 'b', linewidth=0.5)
plt.plot(T, simulateded_data, 'r', linewidth=3)
plt.grid(True)
plt.xlabel("Time[s]")
plt.ylabel("Values")
plt.title("Acceleration data")
plt.legend(["Measured", "Simulated"])


# Difference
# Filter difference
sig_fft = fft(diff_data)
cut_off = 2

n = np.arange(len(sig_fft))  # Length of the Fourier transform
t = len(sig_fft)/Hz  # sampling period
freq = n/t

sig_fft[np.abs(freq) > cut_off] = 0
filtered = ifft(sig_fft)

# Plot filtered on top of unfiltered difference
plt.figure()
plt.plot(T, diff_data, 'r', linewidth=0.5)
plt.plot(T, filtered, 'b', linewidth=1)
plt.grid(True)
plt.xlabel("Time[s]")
plt.ylabel("Values")
plt.legend(["Difference", "Filtered"])
plt.title('Difference\nSimulated data - Measured data'.format())


# Problem 2
# *****************************************************************************
# Perform FFT
spectrum = fft(measured_data)

PlotFFT(spectrum, 'FFT of measured data')


# Problem 3
# *****************************************************************************
diff_data = measured_data-simulateded_data
# Perform FFT
diff_spectrum = fft(diff_data)

PlotFFT(diff_spectrum, 'FFT of measured data - simulated data')


# Extra
# *****************************************************************************
# FFT of simulated data
sim_spectrum = fft(simulateded_data)

PlotFFT(sim_spectrum, 'FFT of simulated data')

print("Done")