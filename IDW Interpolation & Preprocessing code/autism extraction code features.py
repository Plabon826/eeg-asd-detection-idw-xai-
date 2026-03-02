import numpy as np
import pandas as pd
import warnings
import mne
from scipy.signal import butter, filtfilt, periodogram
from scipy.stats import skew, kurtosis

# --- PARAMETERS ---
fs = 500  # Sampling frequency
epoch_duration = 2  # Duration of each epoch in seconds
epoch_samples = fs * epoch_duration  # Number of samples per epoch

# Path to the single .fif file
fif_filepath = '/content/new_channels_added_interpolated_1Abby_Resting.fif'

feature_names = [
    'mean', 'median', 'std', 'skew', 'kurtosis', 'ptp',
    'power_delta', 'power_theta', 'power_alpha', 'power_beta', 'alpha_beta_ratio',
    'hjorth_mobility', 'hjorth_complexity', 'ssc', 'waveform_length',
    'spectral_entropy', 'median_freq', 'peak_freq', 'sample_entropy', 'approx_entropy',
    'perm_entropy', 'hurst_exp', 'corr_dimension', 'dfa',
    'theta_alpha_ratio', 'delta_theta_ratio'
]
num_leads = 62  # Number of leads (channels) - Corrected from 29 to 62
col_names = [f'lead{lead}_{fn}' for lead in range(1, num_leads + 1) for fn in feature_names]
expected_feature_len = len(col_names)

# --- BANDPASS FILTER DESIGN ---
def design_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')
    return b, a

bp_b, bp_a = design_bandpass(0.5, 40, fs)

# --- FEATURE EXTRACTION HELPERS ---
def bandpower(x, fs, freq_band):
    f, Pxx = periodogram(x, fs=fs)
    freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
    return np.trapz(Pxx[freq_mask], f[freq_mask])

def hjorth_params(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    var0 = np.var(signal)
    var1 = np.var(diff1)
    var2 = np.var(diff2)
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if var1 > 0 and mobility > 0 else 0
    return mobility, complexity

def slope_sign_changes(signal):
    return np.sum(np.diff(np.sign(np.diff(signal))) != 0)

def spectral_entropy(signal, fs):
    f, Pxx = periodogram(signal, fs=fs)
    Pxx_norm = Pxx / (np.sum(Pxx) + np.finfo(float).eps)
    se = -np.sum(Pxx_norm * np.log2(Pxx_norm + np.finfo(float).eps))
    return se

def median_frequency(pxx, freqs):
    cumulative_power = np.cumsum(pxx)
    total_power = cumulative_power[-1]
    mfreq = freqs[np.searchsorted(cumulative_power, total_power/2)]
    return mfreq

def sample_entropy(signal, m, r):
    N = len(signal)
    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:,None,:] - x[None,:,:]), axis=2) <= r, axis=0) - 1
        return np.sum(np.log((C + 1e-10) / (N - m)))
    return -(_phi(m+1) - _phi(m))

def approximate_entropy(signal, m, r):
    N = len(signal)
    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:,None,:] - x[None,:,:]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C + np.finfo(float).eps)) / (N - m + 1)
    return _phi(m) - _phi(m+1)

def permutation_entropy(signal, m, delay):
    from math import factorial
    n = len(signal)
    perms_count = factorial(m)
    patterns = np.zeros(perms_count)
    def perm_to_idx(perm):
        idx = 0
        factor = 1
        for i in range(m-1, -1, -1):
            cnt = np.sum(perm[i] > perm[i+1:])
            idx += cnt * factor
            factor *= (m - i)
        return idx
    for i in range(n - delay*(m-1)):
        window = signal[i:i+delay*m:delay]
        idx = perm_to_idx(np.argsort(window))
        patterns[idx] += 1
    patterns /= np.sum(patterns)
    patterns = patterns[patterns > 0]
    return -np.sum(patterns * np.log(patterns))

def hurst_exponent(signal):
    N = len(signal)
    Y = np.cumsum(signal - np.mean(signal))
    R = np.max(Y) - np.min(Y)
    S = np.std(signal)
    if S == 0 or R == 0:
        return 0
    return np.log(R / S) / np.log(N)

def correlation_dimension(signal, tau, m):
    N = len(signal) - (m - 1) * tau
    vectors = np.array([signal[i:i+N] for i in range(0, tau*m, tau)]).T
    dist = pdist(vectors)
    r_vals = np.logspace(np.log10(np.min(dist)+1e-10), np.log10(np.max(dist)), 20)
    C = np.array([np.sum(dist < r) / len(dist) for r in r_vals])
    p = np.polyfit(np.log(r_vals), np.log(C), 1)
    return p[0]

def detrended_fluctuation(signal):
    N = len(signal)
    Y = np.cumsum(signal - np.mean(signal))
    scales = np.floor(np.logspace(np.log10(5), np.log10(N/4), 20)).astype(int)
    F = np.zeros(len(scales))
    for i, s in enumerate(scales):
        ns = N // s
        rms = []
        for v in range(ns):
            idx = slice(v*s, (v+1)*s)
            coeffs = np.polyfit(np.arange(s), Y[idx], 1)
            trend = np.polyval(coeffs, np.arange(s))
            rms.append(np.sqrt(np.mean((Y[idx] - trend) ** 2)))
        F[i] = np.sqrt(np.mean(np.square(rms)))
    p = np.polyfit(np.log(scales), np.log(F), 1)
    return p[0]

# --- Process the single .fif file ---
def process_single_fif(fif_filepath):
    try:
        raw = mne.io.read_raw_fif(fif_filepath, preload=True)
    except ValueError as e:
        warnings.warn(f'Error reading {fif_filepath}: {e}')
        return [], [], []

    # Get the sampling rate from the raw data
    fs = raw.info['sfreq']
    print(f'Processing file: {fif_filepath} (sampling frequency: {fs})')

    # Split data into epochs
    n_epochs = len(raw) // epoch_samples
    print(f'Processing {n_epochs} epochs in file {fif_filepath}')

    local_features = []
    local_labels = []
    local_file_epochs = []

    for ep in range(n_epochs):
        epoch = raw.get_data(start=ep*epoch_samples, stop=(ep+1)*epoch_samples)  # Get raw data for the epoch
        epoch_features = []

        for lead in range(epoch.shape[0]):  # Loop through each lead (channel)
            signal_raw = epoch[lead, :]
            signal = filtfilt(bp_b, bp_a, signal_raw)

            # Compute features (same as in your provided code)
            f_mean = np.mean(signal)
            f_median = np.median(signal)
            f_std = np.std(signal)
            f_skew = skew(signal)
            f_kurt = kurtosis(signal)
            f_ptp = np.ptp(signal)
            f_power_delta = bandpower(signal, fs, [0.5, 4])
            f_power_theta = bandpower(signal, fs, [4, 8])
            f_power_alpha = bandpower(signal, fs, [8, 13])
            f_power_beta = bandpower(signal, fs, [13, 30])
            f_alpha_beta_ratio = f_power_alpha / max(f_power_beta, 1e-10)

            f_hjorth_mobility, f_hjorth_complexity = hjorth_params(signal)
            f_ssc = slope_sign_changes(signal)
            f_wl = np.sum(np.abs(np.diff(signal)))
            f_spectral_entropy = spectral_entropy(signal, fs)
            pxx, freqs = periodogram(signal, fs)
            f_median_freq = median_frequency(pxx, freqs)
            f_peak_freq = freqs[np.argmax(pxx)]
            f_sampen = sample_entropy(signal, 2, 0.2*np.std(signal))
            f_apen = approximate_entropy(signal, 2, 0.2*np.std(signal))
            f_perm_entropy = permutation_entropy(signal, 3, 1)
            f_hurst = hurst_exponent(signal)
            f_corr_dim = correlation_dimension(signal, 10, 2)
            f_dfa = detrended_fluctuation(signal)

            f_theta_alpha_ratio = f_power_theta / max(f_power_alpha, 1e-10)
            f_delta_theta_ratio = f_power_delta / max(f_power_theta, 1e-10)

            lead_features = [f_mean, f_median, f_std, f_skew, f_kurt, f_ptp,
                             f_power_delta, f_power_theta, f_power_alpha, f_power_beta, f_alpha_beta_ratio,
                             f_hjorth_mobility, f_hjorth_complexity, f_ssc, f_wl,
                             f_spectral_entropy, f_median_freq, f_peak_freq, f_sampen, f_apen,
                             f_perm_entropy, f_hurst, f_corr_dim, f_dfa,
                             f_theta_alpha_ratio, f_delta_theta_ratio]

            epoch_features.extend(lead_features)

        if len(epoch_features) != expected_feature_len:
            raise ValueError(f'Feature length mismatch in file {fif_filepath} epoch {ep+1}. Expected {expected_feature_len}, got {len(epoch_features)}')

        local_features.append(epoch_features)
        local_labels.append(0)  # Set label to 0 (or change if needed)
        local_file_epochs.append(f"{fif_filepath}_epoch{ep+1}")

    return local_features, local_labels, local_file_epochs

# --- Process the selected .fif file ---
features_all, labels_all, file_epochs_all = process_single_fif(fif_filepath)

if len(features_all) == 0:
    raise RuntimeError('No features extracted. Check your data and code.')

df = pd.DataFrame(features_all, columns=col_names)
df['Label'] = labels_all
df['File_Epoch'] = file_epochs_all

df.to_csv('eeg_features_single_fif.csv', index=False)
print('Feature extraction done! Saved eeg_features_single_fif.csv')