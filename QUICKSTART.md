# Quick Start Guide for Expression Detector

## Prerequisites
- Windows 10/11 with Python 3.8+
- Webcam or built-in laptop camera
- Modern web browser (Chrome, Edge, Firefox)

## Step-by-Step Setup

### 1. Open Terminal
Navigate to the project folder:
```powershell
cd d:\CODE\code\forfun\expression
```

### 2. Create Virtual Environment (Recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes as it downloads several large packages (TensorFlow, OpenCV, etc.)

### 4. Run the Application
```powershell
python app.py
```

You should see output like:
```
Initializing Expression Detector...
Camera and detector initialized successfully
Starting Flask server on http://localhost:5000
```

### 5. Open in Browser
Open your web browser and go to:
```
http://localhost:5000
```

## Usage

1. **Allow Camera Access**: Your browser may ask for camera permissions - click "Allow"
2. **Position Your Face**: Make sure your face is visible in the camera feed
3. **View Results**: 
   - Green facial landmarks will appear on your face
   - FPS counter shows in the top-right
   - Real-time statistics update on the right panel

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
```powershell
# Make sure virtual environment is activated, then reinstall:
pip install --upgrade opencv-python
```

### Camera not working
- Check Windows Settings → Privacy & Security → Camera
- Ensure the app has camera permissions
- Try a different browser
- Restart the application

### Slow performance
- Close other applications
- Try lowering browser zoom
- Check that CPU usage isn't too high in Task Manager

### Port 5000 already in use
```powershell
# Kill the process using port 5000, or use a different port:
# Edit app.py, change: app.run(port=5001)
```

## To Stop the Server
Press `Ctrl+C` in the terminal

## Next Steps

- Add emotion detection (not currently implemented)
- Add gesture recognition
- Record video clips
- Save snapshots
- Improve performance with GPU acceleration

## System Requirements Explanation

- **Flask**: Web framework for serving the application
- **OpenCV (cv2)**: Computer vision library for camera capture
- **MediaPipe**: Google's framework for facial landmarks detection
- **NumPy**: Numerical computing library
- **TensorFlow/Keras**: Deep learning frameworks (for future emotion detection)

Enjoy your expression detector! 🎬
