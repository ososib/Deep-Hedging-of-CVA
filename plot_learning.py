# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:00:04 2024

@author: osman
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards_over_time(file_path):
    # Read the CSV file, skipping the first row if it contains metadata
    data = pd.read_csv(file_path, skiprows=1)
    
    # Ensure columns are correctly named if there are any issues with extra spaces
    data.columns = data.columns.str.strip()
    #data['cumulative_r'] = data['r'].cumsum()
    
    # Plotting the rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['r'], label='Reward', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def replace_backslash_with_forward_slash(input_string):
    # Replace all backslashes with forward slashes
    return input_string.replace("\\", "/")

# Example usage:
    
path_dir = input ("Enter path :") 
path_dir = replace_backslash_with_forward_slash(path_dir)
plot_rewards_over_time(path_dir)
