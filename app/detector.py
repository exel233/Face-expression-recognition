import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

import mediapipe

legacy_mediapipe_root = Path(mediapipe.__file__).resolve().parent.parent / "~ediapipe"
if legacy_mediapipe_root.exists():
    legacy_path = str(legacy_mediapipe_root)
    if legacy_path not in mediapipe.__path__:
        mediapipe.__path__.append(legacy_path)

try:
    from mediapipe.python.solutions import face_mesh
    from mediapipe.python.solutions import drawing_utils
    from mediapipe.python.solutions import drawing_styles
except Exception as exc:
    raise ImportError(
        "MediaPipe Face Mesh is not available in this environment. "
        "The project found the tasks package, but the legacy solutions API "
        "could not be loaded."
    ) from exc

class EmotionDetector:
    """Detects emotion from facial image with improved accuracy"""
    
    def __init__(self):
        """Initialize emotion detector with enhanced model"""
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = None
        self.emotion_history = []  # For temporal smoothing
        self.history_size = 5  # Keep last 5 detections
        self._init_model()

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize a score dictionary to probabilities."""
        clipped = {emotion: max(0.0, float(value)) for emotion, value in scores.items()}
        total = sum(clipped.values())
        if total <= 0:
            neutral_default = {emotion: 0.0 for emotion in self.emotions}
            neutral_default['Neutral'] = 1.0
            return neutral_default
        return {emotion: value / total for emotion, value in clipped.items()}
    
    def _init_model(self):
        """Initialize the emotion detection model with better architecture"""
        try:
            from tensorflow.keras.models import load_model
            
            # Try to load a pre-trained model if it exists
            model_path = Path(__file__).parent.parent / 'models' / 'emotion_model.h5'
            if model_path.exists():
                try:
                    self.model = load_model(str(model_path))
                    print("Loaded pre-trained emotion model")
                    return
                except Exception as e:
                    print(f"Warning: Model exists but failed to load: {e}")
                    model_path.unlink(missing_ok=True)

            # If pre-trained model is not available, use heuristic only
            print("No pre-trained model available, using heuristic detection")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not initialize emotion model context: {e}")
            self.model = None
    
    def _create_improved_model(self):
        """Create an improved CNN for emotion detection"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            # Improved CNN architecture
            self.model = Sequential([
                Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
                BatchNormalization(),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(128, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                Conv2D(128, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Conv2D(256, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                Conv2D(256, kernel_size=(3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(7, activation='softmax')
            ])
            
            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Created improved emotion detection model")
        except Exception as e:
            print(f"Warning: Could not create improved model: {e}")
            self.model = None
    
    def detect_emotion(self, face_roi: np.ndarray, smile_score: float = 0.0, landmarks: List = None) -> Optional[Dict]:
        """Detect emotion from face region of interest with improved accuracy
        
        Args:
            face_roi: Face region as grayscale image
            smile_score: Value [0.0-1.0] indicating smile intensity (computed from landmarks)
            
        Returns:
            Dict with emotion label, confidence, ranking, and scores
        """
        if face_roi is None:
            return None
        
        all_scores = {emotion: 0.0 for emotion in self.emotions}

        # Preprocess the face image
        processed_face = self._preprocess_face(face_roi)
        heuristic = self._improved_heuristic_emotion_detection(face_roi, landmarks)
        heuristic_scores = heuristic.get('all_predictions', {})

        # Try model-based detection first
        if self.model is not None and processed_face is not None:
            try:
                # Predict
                predictions = self.model.predict(processed_face, verbose=0)
                probs = predictions[0]
                
                # Start from model probabilities
                for i, emotion in enumerate(self.emotions):
                    all_scores[emotion] = float(probs[i])

                # Blend learned probabilities with landmark heuristics.
                # This helps recover expressions like angry/surprise when the raw
                # model is uncertain but the face geometry is more explicit.
                for emotion in self.emotions:
                    model_score = all_scores[emotion]
                    heuristic_score = heuristic_scores.get(emotion, 0.0)
                    all_scores[emotion] = 0.65 * model_score + 0.35 * heuristic_score

                if smile_score > 0.6:
                    all_scores['Happy'] += 0.12
                    all_scores['Fear'] *= 0.7
                    all_scores['Disgust'] *= 0.75
                    all_scores['Angry'] *= 0.85

                all_scores = self._normalize_scores(all_scores)

                # Ranked list
                ranking = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
                top_emotion, top_score = ranking[0]
                
                # Smooth with history
                smoothed = self._temporal_smooth(self.emotions.index(top_emotion), top_score)
                stable_emotion = self.emotions[smoothed['emotion_idx']]
                stable_confidence = smoothed['confidence']

                # If smile score strong, prioritize happy some more
                if smile_score > 0.65 and stable_emotion not in ['Happy', 'Neutral']:
                    stable_emotion = 'Happy'
                    stable_confidence = max(stable_confidence, 0.75)

                return {
                    'emotion': stable_emotion,
                    'confidence': stable_confidence,
                    'all_predictions': all_scores,
                    'ranking': ranking
                }
            except Exception as e:
                print(f"Error with model prediction: {e}")

        # Fallback: improved heuristic-based emotion detection
        heuristic_ranking = sorted(heuristic_scores.items(), key=lambda item: item[1], reverse=True)
        return {
            'emotion': heuristic['emotion'],
            'confidence': heuristic['confidence'],
            'all_predictions': heuristic_scores,
            'ranking': heuristic_ranking
        }
    
    def _preprocess_face(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face image for better emotion detection"""
        try:
            # Ensure grayscale
            if len(face_roi.shape) == 3:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_roi
            
            # Resize to model input size
            face_resized = cv2.resize(face_gray, (48, 48))
            
            # Apply histogram equalization for better contrast
            face_eq = cv2.equalizeHist(face_resized)
            
            # Apply Gaussian blur to reduce noise
            face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)
            
            # Normalize
            face_normalized = face_blur.astype('float32') / 255.0
            face_normalized = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=-1)
            
            return face_normalized
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def _temporal_smooth(self, emotion_idx: int, confidence: float) -> Dict:
        """Apply temporal smoothing to reduce jittery predictions"""
        # Add current detection to history
        self.emotion_history.append({'idx': emotion_idx, 'conf': confidence})
        
        # Keep only recent history
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        # Calculate weighted average
        if len(self.emotion_history) >= 3:
            # Weight recent detections more heavily
            weights = np.linspace(0.5, 1.0, len(self.emotion_history))
            weights = weights / np.sum(weights)
            
            # Calculate weighted emotion index
            emotion_scores = np.zeros(len(self.emotions))
            for i, detection in enumerate(self.emotion_history):
                emotion_scores[detection['idx']] += weights[i] * detection['conf']
            
            smoothed_idx = np.argmax(emotion_scores)
            smoothed_conf = emotion_scores[smoothed_idx]
            
            return {'emotion_idx': smoothed_idx, 'confidence': smoothed_conf}
        else:
            # Not enough history, return current detection
            return {'emotion_idx': emotion_idx, 'confidence': confidence}
    
    def _calculate_landmark_features(self, landmarks: List, face_h: int, face_w: int) -> Dict:
        """Calculate facial features from landmarks for emotion detection"""
        features = {
            'smile_score': 0.0,
            'mouth_corner_up': 0.0,
            'mouth_corner_down': 0.0,
            'eyebrow_lower': 0.0,
            'eyebrow_raise': 0.0,
            'eye_narrow': 0.0,
            'eye_wide': 0.0,
            'brow_furrow': 0.0,
            'mouth_open': 0.0,
            'lip_press': 0.0
        }
        
        if not landmarks or len(landmarks) < 300:
            return features
        
        try:
            face_width = max(1.0, float(face_w))
            face_height = max(1.0, float(face_h))

            # Mouth features
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2.0
            
            mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
            mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))
            
            if mouth_height > 0:
                features['smile_score'] = min(1.0, max(0.0, (mouth_width / mouth_height - 2.0) / 3.0))
            features['mouth_open'] = min(1.0, mouth_height / (face_height * 0.18))
            features['lip_press'] = max(0.0, min(1.0, 1.0 - (mouth_height / (face_height * 0.12))))
            
            # Mouth corner positions relative to mouth center
            left_corner_y = left_mouth[1]
            right_corner_y = right_mouth[1]
            
            avg_corner_y = (left_corner_y + right_corner_y) / 2
            corner_offset = (mouth_center_y - avg_corner_y) / face_height
            features['mouth_corner_up'] = max(0.0, corner_offset * 8.0)
            features['mouth_corner_down'] = max(0.0, -corner_offset * 8.0)
            
            # Eyebrow features using actual brow landmarks.
            left_eyebrow_points = [landmarks[idx] for idx in [70, 63, 105, 66, 107]]
            right_eyebrow_points = [landmarks[idx] for idx in [336, 296, 334, 293, 300]]
            left_eyebrow_inner = landmarks[107]
            right_eyebrow_inner = landmarks[336]
            
            left_eye_upper = landmarks[159]
            left_eye_lower = landmarks[145]
            right_eye_upper = landmarks[386]
            right_eye_lower = landmarks[374]
            
            # Average eyebrow height relative to eyes
            left_eyebrow_avg_y = np.mean([point[1] for point in left_eyebrow_points])
            right_eyebrow_avg_y = np.mean([point[1] for point in right_eyebrow_points])
            left_eye_center_y = (left_eye_upper[1] + left_eye_lower[1]) / 2
            right_eye_center_y = (right_eye_upper[1] + right_eye_lower[1]) / 2
            
            eyebrow_eye_diff_left = left_eyebrow_avg_y - left_eye_center_y
            eyebrow_eye_diff_right = right_eyebrow_avg_y - right_eye_center_y
            avg_eyebrow_diff = (eyebrow_eye_diff_left + eyebrow_eye_diff_right) / 2 / face_height
            
            # Because image y increases downward, larger positive diff means brows are lower.
            features['eyebrow_lower'] = max(0.0, (avg_eyebrow_diff - 0.045) * 8.0)
            features['eyebrow_raise'] = max(0.0, (0.018 - avg_eyebrow_diff) * 12.0)

            inner_brow_distance = np.linalg.norm(np.array(left_eyebrow_inner) - np.array(right_eyebrow_inner))
            furrow_amount = max(0.0, (0.26 - (inner_brow_distance / face_width)) * 6.0)
            features['brow_furrow'] = min(1.0, furrow_amount)
            
            # Eye aspect ratios
            left_eye_width = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[133]))
            left_eye_height = np.linalg.norm(np.array(left_eye_upper) - np.array(left_eye_lower))
            right_eye_width = np.linalg.norm(np.array(landmarks[362]) - np.array(landmarks[263]))
            right_eye_height = np.linalg.norm(np.array(right_eye_upper) - np.array(right_eye_lower))
            
            left_ear = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_ear = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            avg_ear = (left_ear + right_ear) / 2
            
            # Normal EAR is around 0.25-0.35, lower means narrowed, higher means wide.
            features['eye_narrow'] = max(0.0, (0.24 - avg_ear) * 10.0)
            features['eye_wide'] = max(0.0, (avg_ear - 0.34) * 10.0)

        except Exception as e:
            print(f"Error calculating landmark features: {e}")
        
        return features
    
    def _improved_heuristic_emotion_detection(self, face_roi: np.ndarray, landmarks: List = None) -> Dict:
        """Improved heuristic-based emotion detection using facial features and landmarks"""
        try:
            if len(face_roi.shape) == 3:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_roi
            
            # Get face dimensions
            h, w = face_gray.shape
            
            # Divide face into regions for analysis
            # Upper face (eyes, eyebrows)
            upper_region = face_gray[:h//2, :]
            # Lower face (mouth, chin)
            lower_region = face_gray[h//2:, :]
            # Left and right sides
            left_region = face_gray[:, :w//2]
            right_region = face_gray[:, w//2:]
            
            # Calculate features
            upper_brightness = np.mean(upper_region)
            lower_brightness = np.mean(lower_region)
            left_brightness = np.mean(left_region)
            right_brightness = np.mean(right_region)
            
            # Edge detection for expression analysis
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / face_gray.size
            
            # Variance analysis (texture)
            upper_variance = np.var(upper_region)
            # Symmetry analysis
            symmetry = 1.0 - abs(left_brightness - right_brightness) / 255.0
            
            # Landmark-based features
            landmark_features = self._calculate_landmark_features(landmarks, h, w)
            
            # Decision logic based on facial features
            scores = {
                'Happy': 0.0,
                'Sad': 0.0,
                'Angry': 0.0,
                'Surprise': 0.0,
                'Fear': 0.0,
                'Disgust': 0.0,
                'Neutral': 0.0
            }
            
            # Happy
            scores['Happy'] += landmark_features['smile_score'] * 0.95
            scores['Happy'] += landmark_features['mouth_corner_up'] * 0.7
            scores['Happy'] += max(0.0, symmetry - 0.78) * 0.9
            if edge_density < 0.11:
                scores['Happy'] += 0.15

            # Sad
            scores['Sad'] += landmark_features['mouth_corner_down'] * 0.75
            scores['Sad'] += max(0.0, 0.82 - lower_brightness / max(1.0, upper_brightness + 1.0)) * 0.35
            scores['Sad'] += min(0.3, landmark_features['eyebrow_raise'] * 0.25)
            if landmark_features['smile_score'] < 0.12 and landmark_features['mouth_open'] < 0.32:
                scores['Sad'] += 0.12

            # Angry
            scores['Angry'] += landmark_features['eyebrow_lower'] * 0.95
            scores['Angry'] += landmark_features['brow_furrow'] * 1.05
            scores['Angry'] += landmark_features['eye_narrow'] * 0.65
            scores['Angry'] += landmark_features['lip_press'] * 0.45
            if upper_variance > np.var(face_gray) * 1.15:
                scores['Angry'] += 0.12
            if landmark_features['smile_score'] < 0.08:
                scores['Angry'] += 0.08

            # Surprise
            scores['Surprise'] += landmark_features['mouth_open'] * 0.9
            scores['Surprise'] += landmark_features['eyebrow_raise'] * 0.8
            scores['Surprise'] += landmark_features['eye_wide'] * 0.65
            if edge_density > 0.14:
                scores['Surprise'] += 0.12

            # Fear
            scores['Fear'] += landmark_features['eyebrow_raise'] * 0.45
            scores['Fear'] += landmark_features['eye_wide'] * 0.55
            scores['Fear'] += landmark_features['mouth_open'] * 0.35
            if symmetry < 0.8 and edge_density > 0.12:
                scores['Fear'] += 0.12

            # Disgust
            scores['Disgust'] += landmark_features['eye_narrow'] * 0.4
            scores['Disgust'] += landmark_features['brow_furrow'] * 0.35
            if 0.08 < edge_density < 0.16 and symmetry < 0.82:
                scores['Disgust'] += 0.18

            # Neutral
            calm_face = [
                landmark_features['smile_score'],
                landmark_features['mouth_corner_up'],
                landmark_features['mouth_corner_down'],
                landmark_features['eyebrow_lower'],
                landmark_features['eyebrow_raise'],
                landmark_features['eye_narrow'],
                landmark_features['eye_wide'],
                landmark_features['brow_furrow'],
            ]
            if symmetry > 0.86 and abs(upper_brightness - lower_brightness) < 18:
                scores['Neutral'] += 0.3
            if 0.05 < edge_density < 0.12:
                scores['Neutral'] += 0.18
            if max(calm_face) < 0.12 and landmark_features['mouth_open'] < 0.28:
                scores['Neutral'] += 0.45

            scores = self._normalize_scores(scores)

            # Find best emotion
            best_emotion = max(scores, key=scores.get)
            confidence = min(0.9, scores[best_emotion] + 0.1)  # Cap at 90% for heuristics
            
            return {
                'emotion': best_emotion,
                'confidence': confidence,
                'all_predictions': scores
            }
        except Exception as e:
            print(f"Error in improved heuristic detection: {e}")
            return {
                'emotion': 'Unknown',
                'confidence': 0.0,
                'all_predictions': {e: 0.0 for e in self.emotions}
            }


class FacialLandmarksDetector:
    """Detects facial landmarks using MediaPipe"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = face_mesh
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize emotion detector
        self.emotion_detector = EmotionDetector()
        
    def detect_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect facial landmarks and emotion in the image
        
        Returns:
            Dictionary with landmarks data and emotion detection or None if no face found
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w, c = image.shape
        
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        # Get face ROI for emotion detection
        emotion_data = None
        face_roi = self.get_face_roi(image, landmarks)
        smile_score = self.compute_smile_score(landmarks)
        if face_roi is not None and face_roi.size > 0:
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            emotion_data = self.emotion_detector.detect_emotion(face_roi_gray, smile_score=smile_score, landmarks=landmarks)
            
        return {
            'landmarks': landmarks,
            'face_landmarks': face_landmarks,
            'emotion': emotion_data,
            'smile_score': smile_score
        }

    def draw_landmarks(self, image: np.ndarray, landmarks_data: Dict) -> np.ndarray:
        """Draw facial landmarks on the image"""
        if landmarks_data is None:
            return image
        
        face_landmarks = landmarks_data['face_landmarks']
        
        # Draw face mesh using MediaPipe drawing utilities
        self.mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        return image
    
    def compute_smile_score(self, landmarks: List) -> float:
        """Compute a smile strength score from MediaPipe landmarks (0..1)."""
        try:
            if not landmarks or len(landmarks) < 300:
                return 0.0

            # Mouth corner and center points in MediaPipe Face Mesh
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            
            mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
            mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))

            if mouth_height <= 0:
                return 0.0

            ratio = mouth_width / mouth_height
            # Normalize to 0..1 using expected smile range
            score = (ratio - 2.0) / 3.0
            score = max(0.0, min(1.0, score))

            return score
        except Exception:
            return 0.0
    
    def get_face_roi(self, image: np.ndarray, landmarks: List) -> Optional[np.ndarray]:
        """Extract and preprocess face region of interest for better emotion detection"""
        if not landmarks:
            return None
        
        # Get bounding box from landmarks with better margins
        x_coords = [pt[0] for pt in landmarks]
        y_coords = [pt[1] for pt in landmarks]
        
        x_min = max(0, min(x_coords) - 20)  # Larger margin
        x_max = min(image.shape[1], max(x_coords) + 20)
        y_min = max(0, min(y_coords) - 30)  # Extra margin for forehead
        y_max = min(image.shape[0], max(y_coords) + 20)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # Extract face ROI
        face_roi = image[y_min:y_max, x_min:x_max]
        
        # Ensure minimum size
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            return None
        
        # Apply preprocessing for better emotion detection
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_roi
            
            # Apply bilateral filter to reduce noise while keeping edges
            face_filtered = cv2.bilateralFilter(face_gray, 9, 75, 75)
            
            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_enhanced = clahe.apply(face_filtered)
            
            return face_enhanced
        except Exception as e:
            print(f"Error preprocessing face ROI: {e}")
            return face_roi
