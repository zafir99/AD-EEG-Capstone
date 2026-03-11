from os import getcwd, makedirs
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
import numpy as np

cwd = Path(getcwd())
# project root directory
dspath = cwd.parent

out_folder = "out"
out_path = dspath / out_folder
#  paths to the CSV files inside the 'out' folder
con_path = out_path / "con_channel_band_avg.csv"
alz_path = out_path / "alz_channel_band_avg.csv"

if not (con_path.exists() and alz_path.exists()) :
    raise Exception("1 or more .csv files missing! Run datagen.py before running this script!")

print(f"'out' folder found at: {out_path}")
print(f"Control Path: {con_path}")
print(f"Alzheimer Path: {alz_path}")

total_sub = 65
alz_sub = 36
con_sub = 29

#caclulating band power from the PSD data from datagen

def calculate_band_power(data, low, high):
    #filter on the ranges
    band_data = data[(data['Frequency'] >= low) & (data['Frequency'] <= high)]
    return band_data['Power'].mean()



# Load data
alz_data = pd.read_csv(alz_path)
con_data = pd.read_csv(con_path)

#debug statments
print(alz_data.head())
print(alz_data.columns)
print(con_data.head())
print(con_data.columns)

#combine data into one dataframe
combined = pd.concat([alz_data, con_data], ignore_index=True)

bands={
'Delta': (1,4),
'Theta' : (4,8),
'Alpha' : (8,13),
'Beta'  : (13,30)
}
avg = pd.DataFrame(index = range(len(combined)), columns=bands.keys())


#loop through each row (sucject) calculate averages
for index in range(len(combined)):
    subject_data = combined.iloc[[index]]
    for band, (low,high) in bands.items():
        avg.at[index,bands] = calculate_band_power(subject_data,low,high)

#saving
avg.to_csv('avg/sub', index=False)


# Boxplot for Alpha Power
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Alpha_Power', data=combined)
plt.title('Boxplot of Alpha Power (AD vs Control)')
plt.ylabel('Alpha Power (µV²)')
plt.show()

# Boxplot for Theta Power
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Theta_Power',data=combined)
plt.title('Boxplot of Theta Power (AD vs Control)')
plt.ylabel('Theta Power (µV²)')
plt.show()

#calculating average band power for each subject

# calculating mean band power  for each band, seperated by groups
alz_avg = alz_data[['Alpha_Power', 'Theta_Power', 'Delta_Power', 'Beta_Power']].mean().reset_index()
alz_avg.columns = ['Band', 'Mean_Power'] #create columns for it

con_average_power = con_data[['Alpha_Power', 'Theta_Power', 'Delta_Power', 'Beta_Power']].mean().reset_index()
con_average_power.columns = ['Band', 'Mean_Power']

print("AD Group Average Power:\n", alz_avg)
print("Control Group Average Power:\n", con_average_power)


# combine dataframes
#mean_data = pd.concat([alz_avg, con_average_power], ignore_index=True)
alz_data['Group'] ='AD'
con_data['Group'] ='Control'
combined = pd.concat([alz_data, con_data], ignore_index=True)

#print(mean_data.head())
#print(mean_data.columns)

# calculating total mean band power  
alz_avg = alz_data[['Alpha_Power', 'Theta_Power', 'Delta_Power', 'Beta_Power']].mean().reset_index()
alz_avg.columns = ['Band', 'Mean_Power'] #create columns for it

con_average_power = con_data[['Alpha_Power', 'Theta_Power', 'Delta_Power', 'Beta_Power']].mean().reset_index()
con_average_power.columns = ['Band', 'Mean_Power']



# Statistical testing
# Separate data by group
theta_alz = combined[combined['Group'] == 'AD']['Theta_Power']
theta_con = combined[combined['Group'] == 'Control']['Theta_Power']

alpha_alz = combined[combined['Group'] == 'AD']['Alpha_Power']
alpha_con = combined[combined['Group'] == 'Control']['Alpha_Power']

# Perform t-tests
t_stat_alpha, p_value_alpha = ttest_ind(alpha_alz, alpha_con)
t_stat_theta, p_value_theta = ttest_ind(theta_alz, theta_con)

# Output the results
print(f'Alpha Power t-statistic: {t_stat_alpha}, p-value: {p_value_alpha}')
print(f'Theta Power t-statistic: {t_stat_theta}, p-value: {p_value_theta}')