import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

class SignalPreprocessor:
    """Preprocess raw EEG signals for classification"""
    
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def bandpass_filter(self, data, lowcut=1.0, highcut=50.0, order=4):
        """Apply bandpass filter to remove noise"""
        # Normalize frequencies
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    def notch_filter(self, data, freq=50.0, quality=30):
        """Remove powerline interference (50Hz in India)"""
        # Design notch filter
        b, a = signal.iirnotch(freq, quality, self.sample_rate)
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    def normalize(self, data):
        """Normalize signal to zero mean and unit variance"""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return data - mean
        
        normalized = (data - mean) / std
        return normalized
    
    def remove_artifacts(self, data, threshold=3.0):
        """Remove extreme artifacts using threshold"""
        # Calculate z-scores
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        
        # Replace outliers with median
        median = np.median(data)
        cleaned = np.where(z_scores > threshold, median, data)
        
        return cleaned
    
    def preprocess_pipeline(self, raw_signal):
        """Complete preprocessing pipeline"""
        # Step 1: Remove artifacts
        clean_signal = self.remove_artifacts(raw_signal)
        
        # Step 2: Notch filter (remove 50Hz powerline)
        notched = self.notch_filter(clean_signal)
        
        # Step 3: Bandpass filter (keep 1-50Hz range)
        filtered = self.bandpass_filter(notched)
        
        # Step 4: Normalize
        normalized = self.normalize(filtered)
        
        return {
            'raw': raw_signal,
            'processed': normalized,
            'preprocessing_steps': ['artifact_removal', 'notch_filter', 'bandpass_filter', 'normalization']
        }


# Test the preprocessor
if __name__ == "__main__":
    # Generate test signal with noise
    duration = 2.0
    sample_rate = 250
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (10Hz + 20Hz components)
    clean = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Add noise
    noise = np.random.normal(0, 0.5, len(t))
    powerline = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50Hz interference
    
    noisy_signal = clean + noise + powerline
    
    # Preprocess
    preprocessor = SignalPreprocessor(sample_rate)
    result = preprocessor.preprocess_pipeline(noisy_signal)
    
    print("Preprocessing completed!")
    print(f"Raw signal shape: {result['raw'].shape}")
    print(f"Processed signal shape: {result['processed'].shape}")
    print(f"Steps applied: {result['preprocessing_steps']}")
    
    # Show statistics
    print(f"\nRaw signal - Mean: {np.mean(result['raw']):.4f}, Std: {np.std(result['raw']):.4f}")
    print(f"Processed signal - Mean: {np.mean(result['processed']):.4f}, Std: {np.std(result['processed']):.4f}")
