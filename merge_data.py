import pandas as pd

# Load the inputs and the target times
inputs_df = pd.read_csv('InputData.csv')
times_df = pd.read_csv('CPUTimes.csv')

# Combine them row-by-row
inputs_df['cpu_time'] = times_df['CPUTime']

# Save the authentic dataset
inputs_df.to_csv('build_metrics.csv', index=False)
print(f"Successfully merged real data! Shape: {inputs_df.shape}")