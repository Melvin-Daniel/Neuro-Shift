import requests
import sys
sys.path.append('signal-simulator')
from eeg_generator import SyntheticEEGGenerator

# Initialize signal generator
generator = SyntheticEEGGenerator()

print("=" * 60)
print("NEURO-SHIFT - COMPLETE SYSTEM TEST")
print("=" * 60)

# Test different brain commands
test_commands = ['light_on', 'fan_on', 'tv_on', 'light_off', 'fan_off']

for command in test_commands:
    print(f"\n{'='*60}")
    print(f"🧠 Thinking: {command}")
    print("="*60)
    
    # Generate brain signal
    signal_data = generator.generate_command_signal(command, duration=2.0)
    
    # Send to API
    response = requests.post(
        'http://localhost:8000/predict',
        json={
            'signal': signal_data['signal'].tolist(),
            'sample_rate': 250
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Predicted Command: {result['command']}")
        print(f"✓ Confidence: {result['confidence']:.2f}%")
        print(f"✓ Device Action: {result['device_result']['message']}")
        print(f"\nTop 3 Predictions:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for i, (cmd, prob) in enumerate(sorted_probs[:3], 1):
            print(f"  {i}. {cmd}: {prob*100:.2f}%")
    else:
        print(f"✗ Error: {response.status_code}")

# Get final device status
print(f"\n{'='*60}")
print("FINAL DEVICE STATUS")
print("="*60)

response = requests.get('http://localhost:8000/devices')
if response.status_code == 200:
    devices = response.json()['devices']
    for device_name, device_info in devices.items():
        status_emoji = "🟢" if device_info['status'] == 'on' else "⚫"
        print(f"{status_emoji} {device_info['name']}: {device_info['status'].upper()}")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
