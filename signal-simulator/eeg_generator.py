import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

class SyntheticEEGGenerator:
    """Generate synthetic EEG signals simulating brain commands"""
    
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.commands = {
            'light_on': {'alpha': 10, 'beta': 20, 'gamma': 40},
            'light_off': {'alpha': 8, 'beta': 15, 'gamma': 35},
            'fan_on': {'alpha': 12, 'beta': 25, 'gamma': 45},
            'fan_off': {'alpha': 9, 'beta': 18, 'gamma': 38},
            'tv_on': {'alpha': 11, 'beta': 22, 'gamma': 42},
            'idle': {'alpha': 10, 'beta': 20, 'gamma': 40}
        }
    
    def generate_band_signal(self, frequency, duration, amplitude=1.0):
        """Generate signal for specific frequency band"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Add some randomness to simulate natural brain activity
        noise = np.random.normal(0, 0.1, len(t))
        wave = amplitude * np.sin(2 * np.pi * frequency * t) + noise
        return t, wave
    
    def generate_command_signal(self, command='idle', duration=2.0):
        """Generate EEG pattern for specific command"""
        if command not in self.commands:
            command = 'idle'
        
        freq_params = self.commands[command]
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Simulate multi-band EEG (alpha, beta, gamma)
        alpha = np.sin(2 * np.pi * freq_params['alpha'] * t)
        beta = 0.5 * np.sin(2 * np.pi * freq_params['beta'] * t)
        gamma = 0.3 * np.sin(2 * np.pi * freq_params['gamma'] * t)
        
        # Combine bands with noise
        noise = np.random.normal(0, 0.2, len(t))
        eeg_signal = alpha + beta + gamma + noise
        
        return {
            'time': t,
            'signal': eeg_signal,
            'command': command,
            'sample_rate': self.sample_rate
        }
    
    def stream_signal(self, command='idle', duration=2.0):
        """Stream signal data in real-time chunks"""
        data = self.generate_command_signal(command, duration)
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        
        for i in range(0, len(data['signal']), chunk_size):
            chunk = data['signal'][i:i+chunk_size]
            yield {
                'timestamp': time.time(),
                'data': chunk.tolist(),
                'command': command
            }
            time.sleep(0.1)  # Simulate real-time streaming
    
    def visualize_signal(self, command='light_on', duration=2.0):
        """Visualize generated EEG signal"""
        data = self.generate_command_signal(command, duration)
        
        plt.figure(figsize=(12, 4))
        plt.plot(data['time'], data['signal'])
        plt.title(f'Synthetic EEG Signal for Command: {command}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (μV)')
        plt.grid(True)
        plt.savefig(f'../docs/signal_{command}.png')
        plt.show()

# Test the generator
if __name__ == "__main__":
    generator = SyntheticEEGGenerator()
    
    # Test generating different command signals
    commands = ['light_on', 'light_off', 'fan_on', 'idle']
    
    for cmd in commands:
        print(f"Generating signal for: {cmd}")
        generator.visualize_signal(cmd, duration=3.0)
