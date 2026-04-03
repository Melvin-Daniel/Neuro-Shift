import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew

class FeatureExtractor:
    """Extract features from preprocessed EEG signals"""
    
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        
        # Define EEG frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def compute_band_power(self, data, band_name):
        """Compute power in specific frequency band"""
        low, high = self.bands[band_name]
        
        # Compute FFT
        fft_vals = fft(data)
        fft_freq = fftfreq(len(data), 1/self.sample_rate)
        
        # Get positive frequencies only
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_vals = np.abs(fft_vals[positive_freq_idx])
        
        # Find indices in band
        band_idx = np.logical_and(fft_freq >= low, fft_freq <= high)
        
        # Calculate band power
        band_power = np.sum(fft_vals[band_idx] ** 2)
        
        return band_power
    
    def extract_statistical_features(self, data):
        """Extract statistical features from signal"""
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'variance': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'rms': np.sqrt(np.mean(data ** 2)),
            'peak_to_peak': np.max(data) - np.min(data)
        }
        return features
    
    def extract_frequency_features(self, data):
        """Extract frequency domain features"""
        features = {}
        
        # Band powers
        for band_name in self.bands.keys():
            power = self.compute_band_power(data, band_name)
            features[f'{band_name}_power'] = power
        
        # Compute total power
        total_power = sum([features[f'{band}_power'] for band in self.bands.keys()])
        
        # Relative band powers
        for band_name in self.bands.keys():
            if total_power > 0:
                features[f'{band_name}_relative'] = features[f'{band_name}_power'] / total_power
            else:
                features[f'{band_name}_relative'] = 0
        
        # Dominant frequency
        fft_vals = fft(data)
        fft_freq = fftfreq(len(data), 1/self.sample_rate)
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_vals = np.abs(fft_vals[positive_freq_idx])
        
        dominant_freq_idx = np.argmax(fft_vals)
        features['dominant_frequency'] = fft_freq[dominant_freq_idx]
        
        return features
    
    def extract_all_features(self, preprocessed_signal):
        """Extract complete feature set"""
        # Statistical features
        stat_features = self.extract_statistical_features(preprocessed_signal)
        
        # Frequency features
        freq_features = self.extract_frequency_features(preprocessed_signal)
        
        # Combine all features
        all_features = {**stat_features, **freq_features}
        
        return all_features
    
    def features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy vector"""
        return np.array(list(features_dict.values()))


# Test the feature extractor
if __name__ == "__main__":
    # Import signal generator from Day 1
    import sys
    sys.path.append('../signal-simulator')
    from eeg_generator import SyntheticEEGGenerator
    
    # Generate test signal
    generator = SyntheticEEGGenerator()
    signal_data = generator.generate_command_signal('light_on', duration=2.0)
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(signal_data['signal'])
    
    print("Feature Extraction completed!")
    print(f"\nTotal features extracted: {len(features)}")
    print("\nSample features:")
    for i, (key, value) in enumerate(list(features.items())[:10]):
        print(f"  {key}: {value:.4f}")
    
    # Convert to vector
    feature_vector = extractor.features_to_vector(features)
    print(f"\nFeature vector shape: {feature_vector.shape}")
    print(f"Feature vector (first 5 values): {feature_vector[:5]}")
