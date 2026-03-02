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
set_path  = Path("/bin/16840351/54Abby_Resting.set")   # <-- change if needed
fif_path  = Path("/content/54Abby_Resting.fif")        # <-- output location

# -------------------------- 3. Load data -------------------------------------
raw = mne.io.read_raw_eeglab(set_path, preload=True)
channels = raw.info["ch_names"]        # existing channel list
sfreq    = raw.info["sfreq"]           # sampling rate

# -------------------------- 4. Define missing headers ------------------------
missing_channels_to_add = [
    'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5',
    'T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz',
    'Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8',
    'FT8','FC6','FC4','FC2','FCz','C2','C4','C6','T8','TP8','CP6','CP4','CP2',
    'P2','P4','P6','P8','P10','PO8','PO4','O2','T4'
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
    # ----------- left hemisphere -----------
    'Fp1':[-27,83,-3],'AF7':[-51,71,-3],'AF3':[-36,76,24],'F1':[-25,62,56],
    'F3':[-48,59,44],'F5':[-64,55,23],'F7':[-71,51,-3],'FT7':[-83,27,-3],
    'FC5':[-78,30,27],'FC3':[-59,31,56],'FC1':[-33,33,74],'C1':[-34,0,81],
    'C3':[-63,0,61],'C5':[-82,0,31],'T7':[-87,0,-3],'TP7':[-83,-27,-3],
    'CP5':[-78,-30,27],'CP3':[-59,-31,56],'CP1':[-33,-33,74],'P1':[-25,-62,56],
    'P3':[-48,-59,44],'P5':[-64,-55,23],'P7':[-71,-51,-3],'P9':[-64,-47,-37],
    'PO7':[-51,-71,-3],'PO3':[-36,-76,24],'O1':[-27,-83,-3],
    # ----------- midline -------------------
    'Iz':[0,-79,-37],'Oz':[0,-87,-3],'POz':[0,-82,31],'Pz':[0,-63,61],
    'CPz':[0,-34,81],'Cz':[0,0,88],'FCz':[0,34,81],'Fz':[0,63,61],'AFz':[0,82,31],
    'Fpz':[0,87,-3],
    # ----------- right hemisphere ----------
    'Fp2':[27,83,-3],'AF8':[51,71,-3],'AF4':[36,76,24],'F2':[25,62,56],
    'F4':[48,59,44],'F6':[64,55,23],'F8':[71,51,-3],'FT8':[83,27,-3],
    'FC6':[78,30,27],'FC4':[59,31,56],'FC2':[33,33,74],'C2':[34,0,81],
    'C4':[63,0,61],'C6':[82,0,31],'T8':[87,0,-3],'TP8':[83,-27,-3],
    'CP6':[78,-30,27],'CP4':[59,-31,56],'CP2':[33,-33,74],'P2':[25,-62,56],
    'P4':[48,-59,44],'P6':[64,-55,23],'P8':[71,-51,-3],'P10':[64,-47,-37],
    'PO8':[51,-71,-3],'PO4':[36,-76,24],'O2':[27,-83,-3],
    # ----------- temporal extras (if used) --
    'T3':[-87,0,-3],'T4':[87,0,-3],'T5':[-87,1,-3],'T6':[87,2,-3]
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