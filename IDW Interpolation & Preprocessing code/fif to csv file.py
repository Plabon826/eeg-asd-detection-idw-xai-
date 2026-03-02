import mne
import pandas as pd

# Step 1: Load the FIF file
fif_file = '/content/54Abby_Resting.fif'  # Replace with your file path
raw = mne.io.read_raw_fif(fif_file, preload=True)

# Step 2: Get the data and channel names
data, times = raw.get_data(return_times=True)
ch_names = raw.ch_names

# Step 3: Convert to DataFrame
df = pd.DataFrame(data.T, columns=ch_names)
df.insert(0, 'Time (s)', times)

# Step 4: Save to CSV
csv_file = '/content/54Abby_Resting.csv'  # Output filename
df.to_csv(csv_file, index=False)

print(f"EEG data saved to {csv_file}")