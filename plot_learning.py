
import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards_over_time(file_path):
    # Read the CSV file, skipping the first row if it contains metadata
    data = pd.read_csv(file_path, skiprows=1)
    
    # Ensure columns are correctly named if there are any issues with extra spaces
    data.columns = data.columns.str.strip()
    #data['cumulative_r'] = data['r'].cumsum()
    # Calculate the moving average of the rewards
    window_size = 10  # Define the size of the moving average window
    data['moving_average'] = data['r'].rolling(window=window_size).mean()
    
    # Plotting the rewards over time
    plt.figure(figsize=(10, 6))
    plt.scatter(data['l'].cumsum(), data['r'], label='Actual Reward', color='blue', alpha=0.6)
    plt.plot(data['l'].cumsum(), data['moving_average'], label='Moving Average (window={})'.format(window_size), color='red', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title('Rewards and Moving Average Over Time')
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
