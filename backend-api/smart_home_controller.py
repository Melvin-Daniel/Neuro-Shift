import json
from datetime import datetime

class SmartHomeController:
    """Mock smart home device controller"""
    
    def __init__(self):
        self.devices = {
            'light': {
                'name': 'Living Room Light',
                'status': 'off',
                'type': 'light',
                'last_updated': None
            },
            'fan': {
                'name': 'Ceiling Fan',
                'status': 'off',
                'type': 'fan',
                'last_updated': None
            },
            'tv': {
                'name': 'Smart TV',
                'status': 'off',
                'type': 'tv',
                'last_updated': None
            }
        }
        
        self.command_history = []
    
    def parse_command(self, command_string):
        """Parse command string (e.g., 'light_on' -> device='light', action='on')"""
        parts = command_string.rsplit('_', 1)
        
        if len(parts) == 2:
            device, action = parts
            if device in self.devices and action in ['on', 'off']:
                return device, action
        
        return None, None
    
    def control_device(self, command_string):
        """Control device based on command"""
        device, action = self.parse_command(command_string)
        
        if device is None:
            return {
                'success': False,
                'message': f'Invalid command: {command_string}',
                'device': None,
                'action': None
            }
        
        # Update device status
        self.devices[device]['status'] = action
        self.devices[device]['last_updated'] = datetime.now().isoformat()
        
        # Log command
        self.command_history.append({
            'command': command_string,
            'device': device,
            'action': action,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'success': True,
            'message': f'{self.devices[device]["name"]} turned {action}',
            'device': device,
            'action': action,
            'device_info': self.devices[device]
        }
    
    def get_device_status(self, device_name):
        """Get status of specific device"""
        if device_name in self.devices:
            return self.devices[device_name]
        return None
    
    def get_all_devices(self):
        """Get status of all devices"""
        return self.devices
    
    def get_command_history(self, limit=10):
        """Get recent command history"""
        return self.command_history[-limit:]
    
    def reset_all_devices(self):
        """Turn off all devices"""
        for device in self.devices:
            self.devices[device]['status'] = 'off'
            self.devices[device]['last_updated'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'message': 'All devices turned off'
        }


# Test the controller
if __name__ == "__main__":
    controller = SmartHomeController()
    
    print("=" * 60)
    print("SMART HOME CONTROLLER - TEST")
    print("=" * 60)
    
    # Test commands
    test_commands = ['light_on', 'fan_on', 'tv_on', 'light_off', 'invalid_command']
    
    for cmd in test_commands:
        print(f"\nExecuting command: {cmd}")
        result = controller.control_device(cmd)
        print(f"Result: {result['message']}")
        print(f"Success: {result['success']}")
    
    # Show all device statuses
    print("\n" + "=" * 60)
    print("CURRENT DEVICE STATUS")
    print("=" * 60)
    for device_name, device_info in controller.get_all_devices().items():
        print(f"{device_info['name']}: {device_info['status'].upper()}")
    
    # Show command history
    print("\n" + "=" * 60)
    print("COMMAND HISTORY")
    print("=" * 60)
    for entry in controller.get_command_history():
        print(f"{entry['timestamp']}: {entry['command']} -> {entry['action']}")
