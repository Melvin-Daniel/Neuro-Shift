from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import numpy as np

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural-decoder'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'signal-simulator'))

from command_classifier import CommandClassifier
from smart_home_controller import SmartHomeController

# Initialize FastAPI app
app = FastAPI(
    title="Neuro-Shift API",
    description="Brain-Computer Interface for Smart Home Control",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
classifier = CommandClassifier()
smart_home = SmartHomeController()

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'neural-decoder', 'command_classifier_model.pkl')
try:
    classifier.load_model(model_path)
    print("✓ Neural decoder model loaded successfully")
except:
    print("⚠ Warning: Model not found. Please train the model first.")

# Request/Response models
class SignalData(BaseModel):
    signal: List[float]
    sample_rate: Optional[int] = 250
    
class PredictionResponse(BaseModel):
    command: str
    confidence: float
    device_result: dict
    probabilities: dict

class DeviceCommand(BaseModel):
    command: str

class DeviceStatus(BaseModel):
    device: str
    status: str
    name: str
    last_updated: Optional[str]

# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Neuro-Shift API - Brain-Computer Interface for Smart Home",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "predict": "/predict - Process brain signal and control devices",
            "devices": "/devices - Get all device statuses",
            "control": "/control - Manually control devices",
            "history": "/history - Get command history",
            "reset": "/reset - Turn off all devices"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_and_control(signal_data: SignalData):
    """
    Process brain signal, predict command, and control device
    """
    try:
        # Convert signal to numpy array
        signal_array = np.array(signal_data.signal)
        
        # Predict command using neural decoder
        prediction = classifier.predict(signal_array)
        
        # Control device based on prediction
        device_result = smart_home.control_device(prediction['command'])
        
        return {
            "command": prediction['command'],
            "confidence": prediction['confidence'],
            "device_result": device_result,
            "probabilities": prediction['probabilities']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/devices")
async def get_devices():
    """Get status of all smart home devices"""
    return {
        "success": True,
        "devices": smart_home.get_all_devices()
    }

@app.get("/devices/{device_name}")
async def get_device_status(device_name: str):
    """Get status of specific device"""
    device = smart_home.get_device_status(device_name)
    
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device '{device_name}' not found")
    
    return {
        "success": True,
        "device": device
    }

@app.post("/control")
async def manual_control(device_command: DeviceCommand):
    """Manually control device without brain signal"""
    result = smart_home.control_device(device_command.command)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result

@app.get("/history")
async def get_history(limit: int = 10):
    """Get recent command history"""
    return {
        "success": True,
        "history": smart_home.get_command_history(limit)
    }

@app.post("/reset")
async def reset_devices():
    """Turn off all devices"""
    result = smart_home.reset_all_devices()
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier.is_trained,
        "devices_count": len(smart_home.devices)
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("STARTING NEURO-SHIFT API SERVER")
    print("=" * 60)
    print("\n🧠 Neural Decoder: Ready")
    print("🏠 Smart Home Controller: Ready")
    print("\n📡 Server starting on http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
