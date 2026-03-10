from os import getcwd, makedirs
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

cwd = Path(getcwd())
# project root directory
dspath = cwd.parent

out_folder = "out"
out_path = dspath / out_folder
#  paths to the CSV files inside the 'out' folder
con_path = out_path / "con_channel_band_avg.csv"
alz_path = out_path / "alz_channel_band_avg.csv"

if not con_path.exists() or not alz_path.exists() :
    raise Exception("No .csv files found! Run datagen.py before running this script!")

print(f"'out' folder found at: {out_path}")
print(f"Control Path: {con_path}")
print(f"Alzheimer Path: {alz_path}")

# Load data
#need to filter out the columns we are using (alpha & theta)
alz_data = pd.read_csv(alz_path,usecols=['Channel','Alpha_Power','Theta_Power'])
con_data = pd.read_csv(con_path,usecols=['Channel','Alpha_Power','Theta_Power'])

# calculating mean band power 
mean_alz = alz_data.groupby('Channel').agg({'Alpha_Power': 'mean', 'Theta_Power': 'mean'}).reset_index()
mean_alz['Group'] = 'AD'
mean_con = con_data.groupby('Channel').agg({'Alpha_Power': 'mean', 'Theta_Power': 'mean'}).reset_index()
mean_con['Group'] = 'Control'

# combine dataframes
mean_data = pd.concat([mean_alz, mean_con], ignore_index=True)

# Boxplot for Alpha Power
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Alpha_Power', data=mean_data)
plt.title('Boxplot of Alpha Power (AD vs Control)')
plt.ylabel('Alpha Power (µV²)')
plt.show()

# Boxplot for Theta Power
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Theta_Power', data=mean_data)
plt.title('Boxplot of Theta Power (AD vs Control)')
plt.ylabel('Theta Power (µV²)')
plt.show()

# Statistical testing
# Separate data by group
theta_alz = mean_data[mean_data['Group'] == 'AD']['Theta_Power']
theta_con = mean_data[mean_data['Group'] == 'Control']['Theta_Power']

alpha_alz = mean_data[mean_data['Group'] == 'AD']['Alpha_Power']
alpha_con = mean_data[mean_data['Group'] == 'Control']['Alpha_Power']

# Perform t-tests
t_stat_alpha, p_value_alpha = ttest_ind(alpha_alz, alpha_con)
t_stat_theta, p_value_theta = ttest_ind(theta_alz, theta_con)

# Output the results
print(f'Alpha Power t-statistic: {t_stat_alpha}, p-value: {p_value_alpha}')
print(f'Theta Power t-statistic: {t_stat_theta}, p-value: {p_value_theta}')

