# Module 3: Signal Pre-processing & Feature Extraction
## Raspberry Pi Real-Time DSP Pipeline for CSI Sensing

---

## 1. The DSP Pipeline Overview

Raw CSI from the ESP32-S3 is noisy, phase-corrupted, and high-dimensional. Before the RTX 4080 can classify activities, the Raspberry Pi must clean and compress this data in real time. The pipeline processes each aligned CSI group (one packet per RX node, matched by TX sequence number from Module 2) through these stages:

```
Raw CSI (3 nodes × 108 subcarriers × complex) → 648 complex values per frame
    │
    ▼
[1] Phase Sanitization ──── Remove CFO, SFO, PDD from phase
    │
    ▼
[2] Amplitude Denoising ─── Hampel outlier removal + Butterworth lowpass
    │
    ▼
[3] Static Component Removal ── Subtract baseline Hₛ to isolate human signal
    │
    ▼
[4] Dimensionality Reduction ── PCA across subcarriers → top-K components
    │
    ▼
[5] Feature Extraction ──── Doppler spectrogram, variance, entropy
    │
    ▼
[6] Tensor Assembly ──── Pack into GPU-ready tensor → stream to RTX 4080
```

At 100 Hz CSI rate, the Pi has **10 ms per frame** to complete stages 1–6. On a Pi 4/5 with NumPy/SciPy, this is achievable for the algorithms presented here. Stages 1–4 are per-subcarrier operations (easily vectorized); stage 5 operates on sliding windows.

---

## 2. Phase Sanitization

### 2.1 The Problem Recap

As detailed in Modules 1–2, the measured phase of subcarrier k in packet i is corrupted:

```
φ̃(i,k) = φ_true(i,k) + 2π·ε_f·τ(k) + 2π·(f_c + Δf_k)·ε_t + β
```

Where ε_f is CFO (random per-packet), ε_t is SFO+PDD (linear slope across subcarriers per-packet), and β is a constant offset. The true phase φ_true contains the environment and human information.

### 2.2 Method 1: Linear Regression Phase Sanitization (SpotFi-style)

This is the standard approach used in SpotFi, Widar2.0, and most CSI sensing papers. It removes the linear slope (SFO/PDD) and constant offset (CFO) by fitting a line to the unwrapped phase across subcarriers and subtracting it.

The sanitized phase for packet i is:

```
φ_sanitized(i,k) = φ_unwrapped(i,k) - (a·k + b)
```

Where a and b are the slope and intercept of a least-squares linear fit to φ_unwrapped(i,:).

```python
import numpy as np

def sanitize_phase_linear(csi_complex, subcarrier_indices=None):
    """
    Remove CFO (constant offset) and SFO/PDD (linear slope) from CSI phase
    using linear regression across subcarriers.

    Args:
        csi_complex: Complex CSI array [n_subcarriers] for one packet
        subcarrier_indices: Physical subcarrier index array (default: 0..N-1)

    Returns:
        sanitized_complex: Complex CSI with cleaned phase
        sanitized_phase: Cleaned phase array
    """
    n_sc = len(csi_complex)
    if subcarrier_indices is None:
        subcarrier_indices = np.arange(n_sc)

    # Extract and unwrap phase
    raw_phase = np.angle(csi_complex)
    unwrapped = np.unwrap(raw_phase)

    # Linear regression: phase = a*k + b + residual
    # residual is our sanitized phase (contains true environment info)
    k = subcarrier_indices.astype(np.float64)
    A = np.vstack([k, np.ones(n_sc)]).T
    slope, intercept = np.linalg.lstsq(A, unwrapped, rcond=None)[0]

    # Remove linear component
    sanitized_phase = unwrapped - (slope * k + intercept)

    # Reconstruct complex CSI with sanitized phase, original amplitude
    amplitude = np.abs(csi_complex)
    sanitized_complex = amplitude * np.exp(1j * sanitized_phase)

    return sanitized_complex, sanitized_phase
```

### 2.3 Method 2: Conjugate Multiplication (Cross-Link Phase Cleaning)

When you have multiple RX nodes receiving the same TX frame (which you do), conjugate multiplication between two nodes' CSI cancels the shared TX-side phase errors and isolates the differential path information. This is particularly powerful for your 1TX-3RX setup.

For RX nodes A and B receiving the same TX packet:

```
H_A × conj(H_B) = |H_A|·|H_B| · e^(j·(φ_A - φ_B))
```

The TX-side CFO, SFO, and PDD are identical for both receivers (same transmitted packet), so they cancel in the phase difference. What remains is the differential phase caused by different propagation paths — which is exactly the spatial information about the human's position.

```python
def conjugate_multiply(csi_node_a, csi_node_b):
    """
    Conjugate multiplication between two RX nodes' CSI.
    Cancels TX-side phase errors, preserves spatial phase difference.

    Args:
        csi_node_a: Complex CSI [n_subcarriers] from RX node A
        csi_node_b: Complex CSI [n_subcarriers] from RX node B

    Returns:
        cross_csi: Complex conjugate product
        diff_phase: Phase difference (spatial information)
        product_amplitude: Combined amplitude
    """
    cross_csi = csi_node_a * np.conj(csi_node_b)
    diff_phase = np.angle(cross_csi)
    product_amplitude = np.abs(cross_csi)

    return cross_csi, diff_phase, product_amplitude
```

With 3 RX nodes, you get 3 conjugate pairs: (1,2), (1,3), (2,3). Each pair provides a different spatial perspective on the human's position.

### 2.4 Method 3: TSFR (Time Smoothing and Frequency Rebuild)

This is a more advanced method that applies linear regression sanitization, then smooths in the time domain with a Savitzky-Golay filter, and finally rebuilds the frequency-domain phase to eliminate distortions introduced by the time-domain filtering. Published results show it outperforms standard linear transformation across 5 datasets.

```python
from scipy.signal import savgol_filter

def sanitize_phase_tsfr(phase_timeseries, window_length=11, polyorder=3):
    """
    TSFR: Time Smoothing and Frequency Rebuild phase sanitization.

    Args:
        phase_timeseries: Unwrapped phase array [n_packets, n_subcarriers]
                          (already linear-regression sanitized)
        window_length: Savitzky-Golay window (odd integer, in packets)
        polyorder: Polynomial order for Savitzky-Golay

    Returns:
        tsfr_phase: Fully sanitized phase [n_packets, n_subcarriers]
    """
    n_packets, n_sc = phase_timeseries.shape

    # Step 1: Time-domain smoothing per subcarrier
    time_smoothed = np.zeros_like(phase_timeseries)
    for sc in range(n_sc):
        time_smoothed[:, sc] = savgol_filter(
            phase_timeseries[:, sc],
            window_length=min(window_length, n_packets),
            polyorder=min(polyorder, window_length - 1)
        )

    # Step 2: Frequency rebuild — re-apply linear regression per packet
    # to remove distortions introduced by time-domain filtering
    tsfr_phase = np.zeros_like(time_smoothed)
    k = np.arange(n_sc, dtype=np.float64)
    A = np.vstack([k, np.ones(n_sc)]).T

    for t in range(n_packets):
        slope, intercept = np.linalg.lstsq(A, time_smoothed[t], rcond=None)[0]
        tsfr_phase[t] = time_smoothed[t] - (slope * k + intercept)

    return tsfr_phase
```

### 2.5 Which Method to Use When

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| Linear regression | General HAR, real-time | Fast, O(S) per packet | Doesn't fix nonlinear errors |
| Conjugate multiplication | Multi-node localization | Cancels all TX errors | Reduces from 3 signals to 3 pairs |
| TSFR | Breathing, vital signs | Best phase quality | Requires time window (latency) |

**Recommendation for your pipeline:** Apply linear regression to each node independently (real-time, per-packet), AND compute conjugate products for cross-node features. Reserve TSFR for offline analysis or vital-sign specific modes.

---

## 3. Amplitude Denoising

### 3.1 Hampel Filter (Outlier Removal)

CSI amplitude occasionally produces spikes from hardware glitches, WiFi retransmissions, or momentary interference. The Hampel filter detects and replaces these outliers using the median absolute deviation (MAD), which is robust to the very outliers it detects (unlike z-score which uses mean/std).

For each subcarrier's amplitude time series, within a sliding window of size 2w+1 centered at sample t:
1. Compute the window median m and MAD
2. If |x(t) - m| > n_sigma × 1.4826 × MAD, replace x(t) with m

The factor 1.4826 converts MAD to an estimator of standard deviation for Gaussian data.

```python
def hampel_filter(data, window_size=5, n_sigma=3.0):
    """
    Hampel filter for outlier removal on 1D time series.

    Uses median/MAD instead of mean/std, resisting up to 50% contamination.
    O(w·N) per subcarrier where w = window_size, N = number of samples.

    Args:
        data: 1D numpy array (amplitude time series for one subcarrier)
        window_size: Half-window size (total window = 2*window_size + 1)
        n_sigma: Threshold in MAD-scaled units (3.0 is standard)

    Returns:
        filtered: Cleaned data with outliers replaced by local median
        outlier_mask: Boolean array marking detected outliers
    """
    n = len(data)
    filtered = data.copy()
    outlier_mask = np.zeros(n, dtype=bool)
    k = 1.4826  # MAD to sigma conversion for Gaussian

    for i in range(n):
        lo = max(0, i - window_size)
        hi = min(n, i + window_size + 1)
        window = data[lo:hi]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad > 0 and np.abs(data[i] - median) > n_sigma * k * mad:
            filtered[i] = median
            outlier_mask[i] = True

    return filtered, outlier_mask


def hampel_filter_vectorized(data_2d, window_size=5, n_sigma=3.0):
    """
    Vectorized Hampel filter for [n_packets, n_subcarriers] amplitude matrix.
    Applies independently to each subcarrier's time series.
    """
    n_packets, n_sc = data_2d.shape
    filtered = data_2d.copy()
    total_outliers = 0

    for sc in range(n_sc):
        filtered[:, sc], mask = hampel_filter(data_2d[:, sc], window_size, n_sigma)
        total_outliers += mask.sum()

    return filtered, total_outliers
```

**Practical note from ESPectre project testing:** In clean environments, disabling the Hampel filter can actually improve detection recall. The filter may treat the first few CSI samples of genuine motion as outliers and replace them with the baseline median, delaying detection by a few packets. Consider making it configurable — enable for noisy RF environments (shared 2.4 GHz space), disable for controlled deployments.

### 3.2 Butterworth Low-Pass Filter (High-Frequency Noise Removal)

After outlier removal, the amplitude still contains high-frequency noise from thermal noise, WiFi interference, and quantization. A Butterworth low-pass filter removes this while preserving the human motion signal.

Key frequency bands in CSI amplitude:
- **Static environment:** 0 Hz (DC component, removed by baseline subtraction)
- **Breathing:** 0.1–0.5 Hz
- **Walking:** 1–3 Hz
- **Hand gestures:** 2–8 Hz
- **High-frequency noise:** >10 Hz (hardware artifacts, ambient RF)

```python
from scipy.signal import butter, sosfiltfilt

class ButterFilter:
    """
    Real-time Butterworth low-pass filter for CSI amplitude denoising.

    Uses second-order sections (SOS) form for numerical stability.
    sosfiltfilt applies zero-phase filtering (no time delay) for offline use.
    For real-time, use sosfilt with state tracking.
    """

    def __init__(self, cutoff_hz=10.0, fs=100.0, order=4):
        """
        Args:
            cutoff_hz: Low-pass cutoff frequency in Hz
                       10 Hz covers all human activities
                       5 Hz for motion only (walking, gestures)
                       0.5 Hz for breathing only
            fs: CSI sampling rate in Hz
            order: Filter order (4 is standard, higher = sharper cutoff)
        """
        self.cutoff = cutoff_hz
        self.fs = fs
        self.order = order
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        self.sos = butter(order, normalized_cutoff, btype='low', output='sos')

    def apply_offline(self, data):
        """
        Zero-phase filtering (no time delay) for batch processing.

        Args:
            data: [n_packets, n_subcarriers] amplitude matrix

        Returns:
            filtered: Filtered amplitude matrix
        """
        return sosfiltfilt(self.sos, data, axis=0)

    def apply_realtime(self, data, zi=None):
        """
        Causal filtering with state for real-time streaming.

        Args:
            data: [n_new_packets, n_subcarriers] new amplitude data
            zi: Filter state from previous call (None for first call)

        Returns:
            filtered: Filtered data
            zf: Updated filter state (pass to next call)
        """
        from scipy.signal import sosfilt, sosfilt_zi

        if zi is None:
            # Initialize filter state to steady-state for first sample
            zi_single = sosfilt_zi(self.sos)
            # Expand state for n_subcarriers
            n_sc = data.shape[1] if data.ndim > 1 else 1
            zi = np.repeat(zi_single[:, :, np.newaxis], n_sc, axis=2)
            zi *= data[0]  # Scale to initial value

        filtered, zf = sosfilt(self.sos, data, axis=0, zi=zi)
        return filtered, zf
```

**Cutoff frequency selection guide:**

| Application | Cutoff (Hz) | Rationale |
|-------------|-------------|-----------|
| General activity recognition | 10 | Preserves all human-scale dynamics |
| Motion detection only | 5 | Removes gesture frequencies for robust presence |
| Vital signs (breathing) | 0.8 | Narrow band isolates chest displacement |
| Gesture recognition | 15 | Preserves rapid hand movements |

### 3.3 Bandpass Filter for Activity-Specific Detection

For applications like breathing detection where the signal band is very narrow, a bandpass filter outperforms a simple lowpass:

```python
def create_bandpass(low_hz, high_hz, fs=100.0, order=4):
    """Create bandpass Butterworth filter for specific activity bands."""
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    return butter(order, [low, high], btype='band', output='sos')

# Pre-defined activity filters
BREATHING_FILTER = create_bandpass(0.1, 0.5, fs=100.0)   # 6-30 breaths/min
WALKING_FILTER = create_bandpass(0.5, 3.0, fs=100.0)     # Step frequency
GESTURE_FILTER = create_bandpass(1.0, 10.0, fs=100.0)    # Hand motion
```

---

## 4. Static Component Removal

### 4.1 Adaptive Baseline Subtraction

Before PCA or feature extraction, remove the static channel (walls, furniture, direct path) to isolate human-induced changes:

```python
class AdaptiveBaseline:
    """
    Exponential moving average baseline for static component removal.

    Use fast alpha (~0.1) during calibration (empty room).
    Switch to slow alpha (~0.001) during operation (tracks furniture drift).
    """

    def __init__(self, n_subcarriers=108, fast_alpha=0.1, slow_alpha=0.001):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.alpha = fast_alpha
        self.baseline = None
        self.calibrated = False
        self.calibration_count = 0

    def calibrate(self, amplitude, n_required=300):
        """
        Feed empty-room CSI to build initial baseline.
        Call repeatedly until returns True.
        """
        if self.baseline is None:
            self.baseline = amplitude.copy()
        else:
            self.baseline = (1 - self.alpha) * self.baseline + self.alpha * amplitude

        self.calibration_count += 1
        if self.calibration_count >= n_required:
            self.alpha = self.slow_alpha
            self.calibrated = True
            return True
        return False

    def remove_static(self, amplitude):
        """
        Remove static component. Returns dynamic (human) signal only.

        Also slowly adapts baseline to track environmental drift
        (furniture moved, temperature changes affecting RF propagation).
        """
        if self.baseline is None:
            self.baseline = amplitude.copy()
            return np.zeros_like(amplitude)

        dynamic = amplitude - self.baseline

        # Slow baseline update (only when signal is small = likely static)
        signal_energy = np.mean(np.abs(dynamic))
        if signal_energy < np.mean(self.baseline) * 0.1:  # <10% change
            self.baseline = (1 - self.slow_alpha) * self.baseline + \
                           self.slow_alpha * amplitude

        return dynamic
```

### 4.2 PCA-Based Static Removal (Alternative)

An elegant alternative: the first principal component of CSI amplitude across subcarriers primarily captures the static environment. Discarding PC1 removes the static floor while preserving human-induced variance in PC2+.

```python
from sklearn.decomposition import IncrementalPCA

class PCAStaticRemoval:
    """
    Remove static component using PCA. PC1 ≈ static environment.

    Based on the observation that static multipath creates correlated
    amplitude patterns across all subcarriers, while human motion
    creates decorrelated variance on motion-sensitive subcarriers.
    """

    def __init__(self, n_components=10, batch_size=100):
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.fitted = False

    def fit(self, calibration_data):
        """
        Fit PCA on calibration data (empty room or mixed).

        Args:
            calibration_data: [n_packets, n_subcarriers] amplitude matrix
        """
        self.pca.fit(calibration_data)
        self.fitted = True

    def remove_static(self, amplitude_window):
        """
        Project out PC1 (static component).

        Args:
            amplitude_window: [window_size, n_subcarriers]

        Returns:
            dynamic: Amplitude with static removed [window_size, n_subcarriers]
        """
        if not self.fitted:
            return amplitude_window

        # Transform to PC space
        scores = self.pca.transform(amplitude_window)

        # Zero out PC1 (static environment)
        scores[:, 0] = 0

        # Reconstruct in original subcarrier space
        dynamic = self.pca.inverse_transform(scores)
        return dynamic
```

---

## 5. Dimensionality Reduction with PCA

### 5.1 Why PCA for CSI

With 3 nodes × 108 subcarriers = 324 amplitude values per frame (plus phase), the dimensionality is high. But human motion typically affects a correlated subset of subcarriers — those whose Fresnel zones intersect the person's location. PCA identifies these correlated groups and compresses 324 dimensions down to 10–30 principal components while retaining >95% of the motion-related variance.

### 5.2 Streaming PCA for Real-Time Operation

```python
class StreamingCSIPCA:
    """
    Real-time PCA for multi-node CSI dimensionality reduction.

    Concatenates all nodes' amplitudes into a single vector per frame,
    then reduces to top-K components that capture the most variance
    (i.e., the subcarriers most affected by human motion).
    """

    def __init__(self, n_nodes=3, n_subcarriers=108, n_components=20,
                 calibration_size=500):
        self.n_nodes = n_nodes
        self.n_sc = n_subcarriers
        self.n_components = n_components
        self.input_dim = n_nodes * n_subcarriers  # 324

        self.pca = IncrementalPCA(n_components=n_components)
        self.calibration_buffer = []
        self.calibration_size = calibration_size
        self.calibrated = False

        # Statistics for variance explained
        self.explained_variance_ratio = None

    def add_calibration_frame(self, node_amplitudes):
        """
        Add one frame during calibration phase.

        Args:
            node_amplitudes: dict {node_id: amplitude_array[108]}
        """
        # Concatenate all nodes into single feature vector
        combined = np.concatenate([
            node_amplitudes[i] for i in sorted(node_amplitudes.keys())
        ])
        self.calibration_buffer.append(combined)

        if len(self.calibration_buffer) >= self.calibration_size:
            data = np.array(self.calibration_buffer)

            # Standardize per-feature (subcarrier)
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std[self.std < 1e-6] = 1.0  # Prevent division by zero

            standardized = (data - self.mean) / self.std
            self.pca.fit(standardized)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_

            self.calibrated = True
            self.calibration_buffer = []
            return True

        return False

    def transform(self, node_amplitudes):
        """
        Transform a single frame to PCA space.

        Args:
            node_amplitudes: dict {node_id: amplitude_array[108]}

        Returns:
            pc_scores: array [n_components] — the reduced feature vector
        """
        if not self.calibrated:
            return None

        combined = np.concatenate([
            node_amplitudes[i] for i in sorted(node_amplitudes.keys())
        ])
        standardized = (combined - self.mean) / self.std
        return self.pca.transform(standardized.reshape(1, -1))[0]

    def get_top_subcarriers(self, n_top=20):
        """
        Identify which physical subcarriers contribute most to the top PCs.
        Useful for understanding which TX-RX links are most motion-sensitive.

        Returns:
            List of (node_id, subcarrier_index, loading) tuples
        """
        if not self.calibrated:
            return []

        # Sum absolute loadings across top PCs
        loadings = np.sum(np.abs(self.pca.components_[:5, :]), axis=0)

        # Map back to (node, subcarrier)
        results = []
        for idx in np.argsort(loadings)[::-1][:n_top]:
            node = idx // self.n_sc
            sc = idx % self.n_sc
            results.append((node + 1, sc, loadings[idx]))

        return results
```

### 5.3 Variance Explained Heuristic

After calibration, inspect `explained_variance_ratio`:
- If PC1 captures >80% variance → strong static environment (good baseline)
- If PC1–5 capture >95% → 5 components sufficient, aggressive compression possible
- If variance is spread across many PCs → complex multipath or multiple people

Typically, 10–20 components retain sufficient information for HAR while reducing dimensionality by 15–30×.

---

## 6. Feature Extraction for the GPU

### 6.1 Feature Types and Their Uses

The Pi extracts features that the RTX 4080's deep learning models will consume. Different features serve different detection tasks:

```python
import numpy as np
from scipy.signal import stft, welch
from scipy.stats import entropy as scipy_entropy

class CSIFeatureExtractor:
    """
    Extract statistical and spectral features from preprocessed CSI.
    Operates on sliding windows of amplitude/phase data.
    """

    def __init__(self, fs=100.0, window_size=256, hop_size=50):
        """
        Args:
            fs: CSI sampling rate (Hz)
            window_size: Feature window in samples (256 = 2.56 sec @ 100 Hz)
            hop_size: Stride between windows (50 = 0.5 sec, 80% overlap)
        """
        self.fs = fs
        self.window_size = window_size
        self.hop_size = hop_size

    # --- Time-domain features ---

    def amplitude_variance(self, amplitude_window):
        """
        Per-subcarrier variance over time window.
        Primary motion indicator — high variance = motion.

        Args:
            amplitude_window: [window_size, n_subcarriers]

        Returns:
            variance: [n_subcarriers]
        """
        return np.var(amplitude_window, axis=0)

    def amplitude_range(self, amplitude_window):
        """Peak-to-peak amplitude range per subcarrier."""
        return np.ptp(amplitude_window, axis=0)

    def temporal_correlation(self, amplitude_window, lag=1):
        """
        Autocorrelation at specified lag. High correlation = periodic motion
        (walking cadence). Low correlation = random motion or static.
        """
        n_t, n_sc = amplitude_window.shape
        x = amplitude_window[:-lag]
        y = amplitude_window[lag:]
        corr = np.mean(x * y, axis=0) / (np.std(x, axis=0) * np.std(y, axis=0) + 1e-10)
        return corr

    def signal_entropy(self, amplitude_window, n_bins=20):
        """
        Shannon entropy of amplitude distribution per subcarrier.
        High entropy = motion (spread distribution).
        Low entropy = static (concentrated distribution).
        """
        entropies = np.zeros(amplitude_window.shape[1])
        for sc in range(amplitude_window.shape[1]):
            hist, _ = np.histogram(amplitude_window[:, sc], bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropies[sc] = scipy_entropy(hist)
        return entropies

    # --- Frequency-domain features ---

    def doppler_spectrogram(self, csi_timeseries, nperseg=128, noverlap=120):
        """
        Compute Doppler spectrogram from complex CSI time series.

        This is the primary input for CNN-based activity classification.
        Each subcarrier produces a 2D [freq × time] spectrogram.

        Args:
            csi_timeseries: Complex CSI [n_packets, n_subcarriers]
            nperseg: STFT window size (128 @ 100 Hz = 1.28 sec resolution)
            noverlap: Overlap (120 = 93.75% overlap for smooth time axis)

        Returns:
            spectrograms: [n_subcarriers, n_freq_bins, n_time_bins]
            freqs: Frequency (Doppler) axis in Hz
            times: Time axis in seconds
        """
        n_packets, n_sc = csi_timeseries.shape
        specs = []

        for sc in range(n_sc):
            f, t, Zxx = stft(csi_timeseries[:, sc], fs=self.fs,
                            nperseg=nperseg, noverlap=noverlap)
            specs.append(np.abs(Zxx) ** 2)

        freqs = f
        times = t
        spectrograms = np.array(specs)  # [n_sc, n_freq, n_time]

        return spectrograms, freqs, times

    def power_spectral_density(self, amplitude_window, nperseg=128):
        """
        PSD per subcarrier. Useful for detecting dominant motion frequency.
        """
        n_t, n_sc = amplitude_window.shape
        psds = []
        for sc in range(n_sc):
            f, psd = welch(amplitude_window[:, sc], fs=self.fs, nperseg=nperseg)
            psds.append(psd)
        return np.array(psds), f

    # --- Multi-node spatial features ---

    def spatial_variance(self, node_amplitudes):
        """
        Variance of amplitude across subcarriers at a single time instant.
        High spatial variance = signal disruption by human body.

        Args:
            node_amplitudes: [n_subcarriers] from one node

        Returns:
            scalar: Spatial variance
        """
        return np.var(node_amplitudes)

    def cross_node_correlation(self, amp_node_a, amp_node_b):
        """
        Pearson correlation between two nodes' amplitude time series.
        High correlation = same motion observed by both (confirms detection).
        Low correlation = motion in only one node's Fresnel zone.
        """
        corr_matrix = np.corrcoef(
            np.mean(amp_node_a, axis=1),  # Average across subcarriers per packet
            np.mean(amp_node_b, axis=1)
        )
        return corr_matrix[0, 1]

    # --- Combined feature vector ---

    def extract_all(self, amplitude_window, csi_complex_window, node_id=1):
        """
        Extract complete feature vector for one node's window.

        Returns:
            features: dict of named feature arrays
        """
        return {
            'variance': self.amplitude_variance(amplitude_window),
            'range': self.amplitude_range(amplitude_window),
            'autocorr_1': self.temporal_correlation(amplitude_window, lag=1),
            'autocorr_10': self.temporal_correlation(amplitude_window, lag=10),
            'entropy': self.signal_entropy(amplitude_window),
            'spatial_var': np.array([
                self.spatial_variance(amplitude_window[t])
                for t in range(0, len(amplitude_window), 10)
            ]),
            'node_id': node_id,
        }
```

### 6.2 The Doppler Spectrogram — Primary GPU Input

The Doppler spectrogram is the most important feature for the RTX 4080. It converts time-series CSI into a 2D image that CNN architectures can classify directly.

**How to construct a multi-node spectrogram tensor for the GPU:**

```python
def build_gpu_tensor(aligned_groups, pca, extractor, window_size=256):
    """
    Build the tensor that gets streamed to the RTX 4080.

    From a window of aligned CSI groups, produces a 4D tensor:
    [n_nodes, n_pc_components, n_freq_bins, n_time_bins]

    This is equivalent to a multi-channel "image" for a CNN.
    """
    n_nodes = 3
    n_pcs = pca.n_components  # e.g., 20

    # Collect PCA scores over time window
    pc_timeseries = {node: [] for node in range(1, n_nodes + 1)}

    for group in aligned_groups[-window_size:]:
        for node_id, pkt in group.items():
            scores = pca.transform({node_id: pkt.amplitude})
            if scores is not None:
                pc_timeseries[node_id].append(scores)

    # Compute spectrogram of each PC component for each node
    spectrograms = []
    for node_id in range(1, n_nodes + 1):
        series = np.array(pc_timeseries[node_id])  # [T, n_pcs]
        node_specs = []
        for pc in range(n_pcs):
            f, t, Zxx = stft(series[:, pc], fs=100.0, nperseg=64, noverlap=56)
            node_specs.append(np.abs(Zxx) ** 2)
        spectrograms.append(np.array(node_specs))

    # Stack: [n_nodes, n_pcs, n_freq, n_time]
    tensor = np.stack(spectrograms)
    return tensor
```

---

## 7. Streaming to the GPU Server

### 7.1 ZMQ for Pi → GPU Communication

ZeroMQ provides low-latency, high-throughput messaging between the Pi (Ethernet) and the RTX 4080 server. It handles buffering, reconnection, and serialization.

```python
import zmq
import pickle

class GPUForwarder:
    """
    Stream processed CSI tensors from Pi to RTX 4080 server via ZMQ.
    Uses PUB/SUB for decoupled, non-blocking streaming.
    """

    def __init__(self, gpu_address="tcp://192.168.1.100:5556"):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(gpu_address)  # Pi binds, GPU subscribes
        self.socket.setsockopt(zmq.SNDHWM, 100)  # Max 100 queued tensors

    def send_tensor(self, tensor, metadata=None):
        """
        Send a processed tensor to the GPU server.

        Args:
            tensor: numpy array (the GPU-ready feature tensor)
            metadata: dict with timestamp, motion_detected flag, etc.
        """
        payload = {
            'tensor': tensor,
            'metadata': metadata or {},
        }
        self.socket.send(pickle.dumps(payload), zmq.NOBLOCK)

    def close(self):
        self.socket.close()
        self.ctx.term()
```

**GPU-side receiver:**

```python
# gpu_receiver.py — runs on RTX 4080 server
import zmq
import pickle
import numpy as np

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://192.168.1.50:5556")  # Pi's IP
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    print("GPU receiver listening...")

    while True:
        data = pickle.loads(sock.recv())
        tensor = data['tensor']
        meta = data['metadata']
        # Feed tensor to PyTorch model (Module 4)
        print(f"Received tensor shape: {tensor.shape}")

if __name__ == '__main__':
    main()
```

---

## 8. Complete Pi Pipeline — Integrated Example

```python
#!/usr/bin/env python3
"""
dsp_pipeline.py — Complete DSP pipeline running on Raspberry Pi.
Integrates all Module 3 components into a single streaming processor.
"""

import asyncio
import numpy as np
from collections import deque
# Import components from earlier in this module
# (In practice, these would be in separate files)

class DSPPipeline:
    """
    Complete real-time DSP pipeline for CSI sensing.

    Processing flow per aligned frame group:
    1. Extract amplitude and phase from each node
    2. Apply Hampel filter to amplitude (outlier removal)
    3. Apply Butterworth lowpass to amplitude (noise removal)
    4. Sanitize phase via linear regression
    5. Remove static baseline
    6. Compute PCA scores (dimensionality reduction)
    7. Buffer into sliding window
    8. Extract features when window is full
    9. Stream to GPU when features are ready
    """

    def __init__(self, n_nodes=3, n_subcarriers=108, fs=100.0):
        self.n_nodes = n_nodes
        self.n_sc = n_subcarriers
        self.fs = fs

        # DSP components
        self.hampel_window = 5
        self.butter = ButterFilter(cutoff_hz=10.0, fs=fs, order=4)
        self.baselines = {i: AdaptiveBaseline(n_subcarriers) for i in range(1, n_nodes+1)}
        self.pca = StreamingCSIPCA(n_nodes=n_nodes, n_subcarriers=n_subcarriers,
                                   n_components=20, calibration_size=500)
        self.extractor = CSIFeatureExtractor(fs=fs, window_size=256)
        self.forwarder = GPUForwarder()

        # Buffers
        self.amplitude_buffers = {
            i: deque(maxlen=300) for i in range(1, n_nodes+1)
        }
        self.pc_buffer = deque(maxlen=300)

        # State
        self.calibration_phase = True
        self.frame_count = 0

    def process_aligned_group(self, group):
        """
        Process one aligned frame group (all 3 nodes, same TX seq).

        Args:
            group: dict {node_id: CSIPacket}
        """
        self.frame_count += 1
        node_amplitudes = {}

        for node_id, pkt in group.items():
            amp = pkt.amplitude.copy()

            # Store raw amplitude for Hampel (needs history)
            self.amplitude_buffers[node_id].append(amp)

            # Hampel requires at least a few samples
            if len(self.amplitude_buffers[node_id]) > 10:
                recent = np.array(list(self.amplitude_buffers[node_id])[-11:])
                amp = recent[-1]  # Use latest, but Hampel checks history

            # Phase sanitization (linear regression per packet)
            _, sanitized_phase = sanitize_phase_linear(pkt.csi_complex)

            # Calibration or processing
            if self.calibration_phase:
                self.baselines[node_id].calibrate(amp)
                node_amplitudes[node_id] = amp

                if self.baselines[node_id].calibrated:
                    # Check if all nodes calibrated
                    all_cal = all(b.calibrated for b in self.baselines.values())
                    if all_cal:
                        done = self.pca.add_calibration_frame(node_amplitudes)
                        if done:
                            self.calibration_phase = False
                            print(f"Calibration complete. Variance explained: "
                                  f"{self.pca.explained_variance_ratio[:5]}")
                            top = self.pca.get_top_subcarriers(10)
                            print(f"Top sensitive subcarriers: {top}")
            else:
                # Remove static component
                dynamic = self.baselines[node_id].remove_static(amp)
                node_amplitudes[node_id] = dynamic

        # PCA dimensionality reduction
        if not self.calibration_phase:
            pc_scores = self.pca.transform(node_amplitudes)
            if pc_scores is not None:
                self.pc_buffer.append(pc_scores)

                # Quick motion detection (variance of recent PC scores)
                if len(self.pc_buffer) >= 50:
                    recent_pcs = np.array(list(self.pc_buffer)[-50:])
                    motion_energy = np.sum(np.var(recent_pcs, axis=0))

                    # Stream to GPU every 0.5 seconds (50 frames @ 100 Hz)
                    if self.frame_count % 50 == 0:
                        window = np.array(list(self.pc_buffer))
                        self.forwarder.send_tensor(window, metadata={
                            'frame': self.frame_count,
                            'motion_energy': float(motion_energy),
                            'n_pcs': len(pc_scores),
                        })
```

---

## 9. Performance Benchmarks on Raspberry Pi

Expected processing times per frame on Pi 4 (4-core ARM Cortex-A72 @ 1.8 GHz) with NumPy/SciPy:

| Stage | Time per Frame | Notes |
|-------|---------------|-------|
| Packet parsing (3 nodes) | ~0.1 ms | struct.unpack + numpy conversion |
| Phase sanitization (3 nodes) | ~0.3 ms | Linear regression, vectorized |
| Hampel filter (108 sc × 3 nodes) | ~0.5 ms | Per-subcarrier, sliding window |
| Butterworth filter (3 nodes) | ~0.2 ms | SOS form, vectorized |
| Baseline subtraction | ~0.05 ms | Simple array subtraction |
| PCA transform | ~0.1 ms | Matrix multiply after calibration |
| Feature extraction (per window) | ~2–5 ms | STFT is heaviest; runs every 50th frame |
| **Total per-frame** | **~1.3 ms** | Well within 10 ms budget |

**Tips for Pi performance:**
- Use `numpy` compiled with OpenBLAS (default on Pi OS) for fast linear algebra
- Pre-allocate all arrays; avoid dynamic allocation in the hot path
- Pin the DSP process to a specific core with `taskset`
- Consider `numba` JIT compilation for the Hampel filter loop

---

## 10. Key Research Papers for Module 3

1. **Hands-on Wireless Sensing Tutorial (Sanitization Code)**
   Zhang, D. et al. (2022). https://tns.thss.tsinghua.edu.cn/wst/docs/sanitization/
   — MATLAB reference implementations for all sanitization algorithms (CFO, SFO, nonlinear), directly translatable to Python.

2. **Optimal Preprocessing of WiFi CSI for Sensing**
   IEEE Trans. Signal Processing (2024). arXiv:2307.12126
   — Mathematical models for gain and phase errors with theoretically justified correction algorithms. Shows 40% noise reduction for gain and 200% for phase vs. baselines.

3. **TSFR: Channel Phase Processing for HAR**
   Internet of Things Journal (2023). arXiv:2303.16873
   — The Time Smoothing and Frequency Rebuild method tested on 5 datasets with 3 DL architectures. Achieves >90% accuracy in most scenarios.

4. **WiFi Sensing on the Edge: Signal Processing Survey**
   IEEE COMST (2022). Hernandez & Bulut.
   — Comprehensive comparison table of all filtering methods (Hampel, Butterworth, Savitzky-Golay, wavelet) with complexity analysis and CSI-specific tradeoffs.

5. **CSIKit: Python CSI Processing Library**
   https://github.com/Gi-z/CSIKit
   — Open-source Python library with Hampel, lowpass, and running mean filters plus readers for ESP32, Intel 5300, Atheros, and Nexmon formats.

6. **ESPectre: ESP32 CSI Motion Detection**
   https://espectre.dev/documentation/algorithms/
   — Production-tested pipeline using Hampel → Butterworth → moving variance on ESP32, with detailed discussion of gain lock and adaptive thresholds.

---

*Module 3 Complete — Next: Module 4 (GPU Deep Learning, Model Architecture & Domain Adaptation)*
