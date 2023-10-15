# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:02:16 2023

@author: magnu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

# Load data
# root = tk.Tk()
# housing_filepath = filedialog.askopenfilename()
# root.destroy()
housing_filepath = 'C:/Users/magnu\/OneDrive - USN/Machine Learning and Sensor Technology/Assignment 2/Housing.csv'

housing_data = pd.read_csv(housing_filepath)
housing_data.info()
print(" Data loaded ")

# Explore the data
print(housing_data.describe())

sns.pairplot(housing_data, vars=['area', 'bedrooms', 'price'], diag_kind='kde')
plt.figure(2)

# Convert objects to numerical
# Replace yes and no with 1 and 0
replacementsYN = {'yes' : 1, 'no' : 0}
catFeatures = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for feature in catFeatures:
    housing_data[feature] = housing_data[feature].map(replacementsYN)
# Replace the furnishing status
replaceFur = {'furnished' : 2, 'semi-furnished' : 1, 'unfurnished' : 0}
housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map(replaceFur).fillna(housing_data['furnishingstatus'])

print(housing_data)

print('Data prepared\n')

# Calculate correlation
correlationMatrix = housing_data.corr()['price']
print(correlationMatrix)

plt.plot(correlationMatrix)
plt.grid()
plt.title('Correlation to price')
print('Finished')
plt.show()
