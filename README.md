# Expression Detector

A real-time facial expression and emotion detector using laptop's camera.

## Features
- **Real-time emotion detection**: Detects happy, sad, angry, surprised, feared, disgusted, and neutral expressions
- **Facial landmarks**: Identifies key facial features (eyes, nose, mouth, etc.)
- **Web interface**: Access the application through a modern web browser
- **Live camera feed**: See the processed video with overlays in real-time

## Requirements
- Python 3.8+
- Webcam/Laptop camera
- Modern web browser (Chrome, Firefox, Edge, Safari)

## Installation

1. Clone or navigate to the project directory:
```bash
cd expression
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Allow camera access when prompted

4. The application will start displaying:
   - Live camera feed with facial landmarks overlaid
   - Detected emotion with confidence score
   - Facial coordinate information

## How It Works

- **Camera Capture**: OpenCV captures video from your webcam
- **Face Detection**: MediaPipe detects faces and facial landmarks
- **Emotion Detection**: A pre-trained deep learning model predicts the emotion
- **Real-time Streaming**: MJPEG stream sends processed frames to the browser

## Project Structure

```
expression/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── app/
│   ├── __init__.py
│   ├── camera.py         # Camera handling and frame processing
│   └── detector.py       # Expression detection logic
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
└── models/               # Pre-trained models (downloaded automatically)
```

## Notes
- First run may take a few minutes as models are downloaded
- Performance depends on your laptop's CPU
- For better performance, consider using GPU support (requires CUDA/cuDNN)

## Troubleshooting

**Camera not working?**
- Check browser permissions for camera access
- Try using HTTPS if on localhost

**Models not downloading?**
- Check internet connection
- Models folder requires write permissions

**Slow performance?**
- Reduce video resolution
- Close other applications
- Consider reducing frame rate

## License
MIT
