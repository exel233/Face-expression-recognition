from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from app.camera import CameraCapture, FrameProcessor, FrameEncoder
from app.detector import FacialLandmarksDetector
import threading
import time
from download_model import download_emotion_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global objects
camera = None
detector = None
lock = threading.Lock()
frame_data = {
    'timestamp': None,
    'landmarks_count': 0,
    'processing_time': 0,
    'emotion': None,
    'confidence': 0,
    'emotion_stable': False,
    'emotion_history': [],
    'emotion_ranking': []
}

def init_camera_and_detector():
    """Initialize camera and detector"""
    global camera, detector
    try:
        # Download emotion detection model
        print("Checking for emotion detection model...")
        download_emotion_model()
        
        camera = CameraCapture()
        camera.start()
        detector = FacialLandmarksDetector()
        print("Camera and detector initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Process a single frame and return annotated frame with data"""
    global frame_data
    
    start_time = time.time()
    
    # Detect landmarks and emotion
    landmarks_data = detector.detect_landmarks(frame)
    
    # Draw landmarks
    if landmarks_data:
        frame = detector.draw_landmarks(frame, landmarks_data)
        landmarks_count = len(landmarks_data['landmarks'])
        
        # Display emotion if detected
        emotion_data = landmarks_data.get('emotion')
        if emotion_data:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            # Color code emotions
            emotion_colors = {
                'Happy': (0, 255, 0),      # Green
                'Sad': (255, 0, 0),        # Blue
                'Angry': (0, 0, 255),      # Red
                'Surprise': (0, 255, 255), # Yellow
                'Fear': (255, 0, 255),     # Magenta
                'Disgust': (128, 128, 0),  # Olive
                'Neutral': (128, 128, 128) # Gray
            }
            
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw emotion on frame with colored background
            emotion_text = f"Emotion: {emotion} ({confidence:.1%})"
            (text_width, text_height), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw background rectangle
            cv2.rectangle(frame, (18, 58), (22 + text_width, 62 + text_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 60), (20 + text_width, 60 + text_height), color, 2)
            
            # Draw text
            cv2.putText(frame, emotion_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw confidence bar with color
            bar_width = int(confidence * 200)
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 100), color, -1)
            cv2.rectangle(frame, (20, 90), (220, 100), color, 2)
            
            with lock:
                frame_data['emotion'] = emotion
                frame_data['confidence'] = confidence
                frame_data['emotion_ranking'] = emotion_data.get('ranking', [])
                
                # Track emotion stability
                frame_data['emotion_history'].append(emotion)
                if len(frame_data['emotion_history']) > 10:
                    frame_data['emotion_history'].pop(0)
                
                # Check if emotion is stable (same emotion in last 5 frames)
                recent_emotions = frame_data['emotion_history'][-5:]
                frame_data['emotion_stable'] = len(set(recent_emotions)) == 1 and len(recent_emotions) >= 5
        else:
            with lock:
                frame_data['emotion'] = None
                frame_data['confidence'] = 0
                frame_data['emotion_ranking'] = []
    else:
        landmarks_count = 0
        # Add "No face detected" message
        frame = FrameProcessor.add_text_overlay(
            frame, "No face detected - Please ensure your face is visible",
            (20, 50), (0, 0, 255), 0.8, 2
        )
    
    # Add FPS and info
    frame = FrameProcessor.add_fps(frame, int(camera.fps) if camera else 0)
    
    processing_time = time.time() - start_time
    
    with lock:
        frame_data['timestamp'] = time.time()
        frame_data['landmarks_count'] = landmarks_count
        frame_data['processing_time'] = processing_time * 1000  # Convert to ms
    
    return frame, {'landmarks': landmarks_count}


def generate_frames():
    """Generator function for streaming frames"""
    while True:
        if not camera or not camera.is_active():
            time.sleep(0.1)
            continue
        
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Process frame
        processed_frame, _ = process_frame(frame)
        
        # Encode frame
        jpeg_data = FrameEncoder.encode_frame(processed_frame, quality=85)
        
        if jpeg_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg_data)).encode() + b'\r\n\r\n'
                   + jpeg_data + b'\r\n')
        
        time.sleep(0.01)  # Limit to ~100 FPS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def get_stats():
    """Get current frame processing stats"""
    with lock:
        emotion = frame_data.get('emotion')
        confidence = frame_data.get('confidence', 0)
        emotion_stable = frame_data.get('emotion_stable', False)
        ranking = frame_data.get('emotion_ranking', [])

        return jsonify({
            'fps': int(camera.fps) if camera else 0,
            'landmarks_detected': frame_data['landmarks_count'],
            'processing_time_ms': round(frame_data['processing_time'], 2),
            'camera_active': camera.is_active() if camera else False,
            'emotion': emotion,
            'confidence': confidence,
            'emotion_stable': emotion_stable,
            'emotion_ranking': ranking
        })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_active': camera.is_active() if camera else False
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Initializing Expression Detector...")
    if init_camera_and_detector():
        print("Starting Flask server on http://localhost:5000")
        try:
            app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if camera:
                camera.stop()
    else:
        print("Failed to initialize camera. Please check your webcam connection.")
