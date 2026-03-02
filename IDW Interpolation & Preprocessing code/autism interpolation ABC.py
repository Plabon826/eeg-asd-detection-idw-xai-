"""
Interpolate missing EEG channels with **Inverse-Distance Weighting (IDW)**

• Starts from an EEGLAB .set file
• Adds any header-only channels that are absent
• Uses hand-entered 3-D locations (10-10 system, in mm) for all 64 electrodes
• Performs IDW interpolation (power = 2) across *all* time points at once
• Saves result as a .fif file, then reloads and prints basic info
"""

# -------------------------- 1. Imports ---------------------------------------
import numpy as np
import mne
from pathlib import Path

# -------------------------- 2. File paths ------------------------------------
set_path  = Path("/bin/16840351/39Abby_Resting.set")   # <-- change if needed
fif_path  = Path("/content/39Abby_Resting.fif")        # <-- output location

# -------------------------- 3. Load data -------------------------------------
raw = mne.io.read_raw_eeglab(set_path, preload=True)
channels = raw.info["ch_names"]        # existing channel list
sfreq    = raw.info["sfreq"]           # sampling rate

# -------------------------- 4. Define missing headers ------------------------
missing_channels_to_add = [
    'A3','A5','A7','A10','A12','A15','A17','A19','A21','A23','A25','A28','A30','A32',
    'B2','B4','B7','B9','B11','B12','B14','B16','B18','B20','B22','B24','B26','B27','B29','B31',
    'C2','C4','C5','C7','C8','C12','C14','C16','C17','C19','C21','C23','C25','C27','C29','C30',
    'D2','D4','D5','D7','D8','D10','D12','D14','D16','D19','D21','D23','D24','D26','D28','D30','D31','D32'
]
# Header-only channels that are *not* in the raw file
channels_to_add = [ch for ch in missing_channels_to_add if ch not in channels]

# -------------------------- 5. Add empty channels ----------------------------
if channels_to_add:
    print("Adding header-only channels:", channels_to_add)

    empty_data  = np.zeros((len(channels_to_add), raw.n_times))
    info_empty  = mne.create_info(channels_to_add, sfreq=sfreq, ch_types=["eeg"]*len(channels_to_add))
    raw_empty   = mne.io.RawArray(empty_data, info_empty)
    raw.add_channels([raw_empty])

# Combined data array (existing + newly added)
combined_data = raw.get_data()

# -------------------------- 6. Electrode locations ---------------------------
manual_positions = {
        'A3': [-27, 83, -3], 'A5': [-51, 71, -3], 'A7': [-36, 76, 24], 'A10': [-25, 62, 56],
        'A12': [-48, 59, 44], 'A15': [-64, 55, 23], 'A17': [-71, 51, -3], 'A19': [-83, 27, -3],
        'A21': [-78, 30, 27], 'A23': [-59, 31, 56], 'A25': [-33, 33, 74], 'A28': [-34, 0, 81],
        'A30': [-63, 0, 61], 'A32': [-82, 0, 31], 'B2': [-87, 0, -3], 'B4': [-83, -27, -3],
        'B7': [-78, -30, 27], 'B9': [-59, -31, 56], 'B11': [-33, -33, 74], 'B12': [-25, -62, 56],
        'B14': [-48, -59, 44], 'B16': [-64, -55, 23], 'B18': [-71, -51, -3], 'B20': [-64, -47, -37],
        'B22': [-51, -71, -3], 'B24': [-36, -76, 24], 'B26': [-27, -83, -3], 'B27': [4.85979E-15, -79, -37],
        'B29': [5.35892E-15, -87, -3], 'B31': [5.00603E-15, -82, 31], 'C2': [3.85723E-15, -63, 61],
        'C4': [2.09517E-15, -34, 81], 'C5': [5.35892E-15, 87, -3], 'C7': [27, 83, -3],
        'C8': [51, 71, -3], 'C12': [36, 76, 24], 'C14': [5.00603E-15, 82, 31], 'C16': [3.85723E-15, 63, 61],
        'C17': [25, 62, 56], 'C19': [48, 59, 44], 'C21': [64, 55, 23], 'C23': [71, 51, -3],
        'C25': [83, 27, -3], 'C27': [78, 30, 27], 'C29': [59, 31, 56], 'C30': [33, 33, 74],
        'D2': [2.09517E-15, 34, 81], 'D4': [34, 0, 81], 'D5': [63, 0, 61],
        'D7': [82, 0, 31], 'D8': [87, 0, -3], 'D10': [83, -27, -3], 'D12': [78, -30, 27],
        'D14': [59, -31, 56], 'D16': [33, -33, 74], 'D19': [25, -62, 56], 'D21': [48, -59, 44],
        'D23': [64, -55, 23], 'D24': [71, -51, -3], 'D26': [64, -47, -37], 'D28': [51, -71, -3],
        'D30': [36, -76, 24], 'D31': [27, -83, -3], 'Cz': [0, 0, 88], 'T3': [-87, 0, -3], 'D32': [87, 0, -3], 'T5': [87, 1, -3], 'T6': [87, 2, -3]
    }
# -------------------------- 7. Split existing vs missing ---------------------
channels_with_coords = [ch for ch in raw.info["ch_names"] if ch in manual_positions]

existing_names = [ch for ch in channels_with_coords if ch not in channels_to_add]
missing_names  = [ch for ch in channels_to_add       if ch in manual_positions]

existing_idx   = [raw.info["ch_names"].index(ch) for ch in existing_names]
missing_idx    = [raw.info["ch_names"].index(ch) for ch in missing_names]

existing_pos   = np.array([manual_positions[ch] for ch in existing_names])
missing_pos    = np.array([manual_positions[ch] for ch in missing_names])

existing_data  = combined_data[existing_idx, :]

# -------------------------- 8. IDW function ----------------------------------
def idw_interpolate(src_pos, src_val, tgt_pos, power=2):
    """Inverse-Distance Weighting interpolation (vectorised)."""
    d = np.linalg.norm(src_pos[:, None, :] - tgt_pos[None, :, :], axis=2)
    d[d == 0] = 1e-12                      # avoid /0
    w = 1.0 / np.power(d, power)           # weights
    w /= w.sum(axis=0, keepdims=True)      # normalise per target
    return (w.T @ src_val)                 # (n_tgt, n_times)

# -------------------------- 9. Interpolate missing --------------------------
if missing_names:
    missing_data = idw_interpolate(existing_pos, existing_data, missing_pos, power=2)
    for row, ch in zip(missing_data, missing_names):
        combined_data[raw.info["ch_names"].index(ch), :] = row

    raw._data = combined_data
    print(f"Filled {len(missing_names)} missing channels via IDW.")

else:
    print("No missing channels needed interpolation.")

# -------------------------- 10. Save as .fif ---------------------------------
raw.save(fif_path, overwrite=True)
print(f"Saved interpolated file to {fif_path}")

# -------------------------- 11. Quick verification ---------------------------
raw2 = mne.io.read_raw_fif(fif_path, preload=False)

print("\n=== FILE SUMMARY ===")
print(f"Channels    : {len(raw2.info['ch_names'])}")
print(f"Sampling Hz : {raw2.info['sfreq']}")


# Optional quick visual check
# raw2.plot(duration=10, n_channels=30)  # uncomment if running interactively