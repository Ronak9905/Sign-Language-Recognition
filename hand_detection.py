import cv2
import mediapipe as mp
from typing import Tuple
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5):
        """
        Initialize Hand Detection with MediaPipe.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Detects hand landmarks in a frame and returns an annotated copy.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            A tuple containing:
            - annotated_frame: A copy of the frame with hand landmarks drawn.
            - result: The raw hand detection result object from MediaPipe.
        """
        # Work on a copy to avoid modifying the original frame
        annotated_frame = frame.copy()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Performance optimization

        # Process frame
        result = self.hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )

        return annotated_frame, result
    
    def close(self):
        """Close the hand detector"""
        self.hands.close()
