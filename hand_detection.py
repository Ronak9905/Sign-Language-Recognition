import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7):
        """
        Initialize Hand Detection with MediaPipe
        
        Args:
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Confidence threshold for detection (0-1)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hands(self, frame):
        """
        Detect hand landmarks in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            frame: Frame with hand landmarks drawn
            result: Hand detection result object
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        result = self.hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
        
        return frame, result
    
    def get_landmarks(self, result):
        """
        Extract hand landmarks from detection result
        
        Args:
            result: Hand detection result object
            
        Returns:
            list: List of hand landmarks (None if no hands detected)
        """
        if result.multi_hand_landmarks:
            return result.multi_hand_landmarks
        return None
    
    def close(self):
        """Close the hand detector"""
        self.hands.close()
