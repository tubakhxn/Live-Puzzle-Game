# Live Puzzle Game (YOLO + OpenCV)
# Entry point for the application



import cv2
import numpy as np
import mediapipe as mp
from puzzle_ui import PuzzleUI

import mediapipe as mp
# --- MediaPipe Hand Detection (v0.10.x+ fallback) ---

def detect_hands_mediapipe(frame):
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_landmarks = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = []
                for lm in handLms.landmark:
                    h, w, _ = frame.shape
                    lm_list.append((int(lm.x * w), int(lm.y * h)))
                hand_landmarks.append(lm_list)
        return hand_landmarks
    except AttributeError:
        # For mediapipe >=0.10.x, use mp.tasks.vision.HandLandmarker
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        import tempfile
        import urllib.request
        import os
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        MODEL_PATH = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)
        hand_landmarks = []
        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand in result.hand_landmarks:
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                hand_landmarks.append(lm_list)
        return hand_landmarks

# --- Main Loop ---

def main():
    cap = cv2.VideoCapture(0)
    captured = False
    puzzle_ui = None
    grid_size = 3
    hand_wide_counter = 0
    hand_wide_required = 3
    
    while True:
        ret, raw_frame = cap.read()

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            h, w = raw_frame.shape[:2]
            dark_bg = np.zeros_like(raw_frame)
            min_dim = min(h, w)
            cam_img = cv2.resize(raw_frame, (min_dim, min_dim))
            dark_bg[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2] = cam_img
            frame = dark_bg.copy()

            if not captured:
                border_color = (57,255,20)
                border_thickness = 2
                title = "LIVE PUZZLE"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, font_thickness)
                cv2.putText(frame, title, ((w-text_w)//2, 35), font, font_scale, border_color, font_thickness, cv2.LINE_AA)
                hand_landmarks = detect_hands_mediapipe(frame)
                # Draw hand skeletons
                for lm_list in hand_landmarks:
                    for idx, (x, y) in enumerate(lm_list):
                        cv2.circle(frame, (x, y), 3, (57,255,20), -1)
                    palm = [0, 1, 2, 5, 9, 13, 17, 0]
                    for i in range(len(palm)-1):
                        cv2.line(frame, lm_list[palm[i]], lm_list[palm[i+1]], (57,255,20), 2)
                    for f in [4, 8, 12, 16, 20]:
                        cv2.line(frame, lm_list[0], lm_list[f], (57,255,20), 1)
                # If two hands, draw rectangle between them
                rect_pts = None
                if len(hand_landmarks) == 2:
                    # Use wrist (0) and pinky (17) of both hands for rectangle
                    pts = [hand_landmarks[0][0], hand_landmarks[0][17], hand_landmarks[1][17], hand_landmarks[1][0]]
                    pts_np = np.array(pts, dtype=np.int32)
                    x_min = np.min(pts_np[:,0])
                    y_min = np.min(pts_np[:,1])
                    x_max = np.max(pts_np[:,0])
                    y_max = np.max(pts_np[:,1])
                    rect_pts = (x_min, y_min, x_max, y_max)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (180,255,80), 2)
                    # Show hint
                    cv2.putText(frame, "PINCH TO CAPTURE", (x_min+5, y_min-10), font, 0.6, (180,255,80), 2)
                # Pinch with either hand to capture
                pinch = False
                for lm in hand_landmarks:
                    x4, y4 = lm[4]
                    x8, y8 = lm[8]
                    if np.hypot(x4-x8, y4-y8) < 40:
                        pinch = True
                if rect_pts and pinch:
                    # Always center the puzzle in the frame after capture
                    rect_w = rect_pts[2] - rect_pts[0]
                    rect_h = rect_pts[3] - rect_pts[1]
                    # Crop from the original camera frame, not dark_bg
                    puzzle_img = raw_frame[rect_pts[1]:rect_pts[3], rect_pts[0]:rect_pts[2]].copy()
                    captured = True
                    puzzle_ui = PuzzleUI(puzzle_img, grid_size)
                    cv2.namedWindow("Puzzle")
                    cv2.setMouseCallback("Puzzle", puzzle_ui.mouse_event)
                cv2.imshow("Live Puzzle", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                # Show puzzle in its own window for mouse solving
                solved = puzzle_ui.show(dark_theme=False, window_name="Puzzle")
                if solved:
                    # Custom overlay for completion
                    solved_overlay = np.zeros_like(puzzle_ui.window)
                    solved_overlay[:] = (20, 20, 20)  # dark background
                    # Trophy icon (simple)
             
                    cv2.putText(solved_overlay, "COMPLETE!", (puzzle_ui.window.shape[1]//2-170, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
                    cv2.putText(solved_overlay, "Tuba", (puzzle_ui.window.shape[1]//2-250, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
                    name = "Tuba"
                    input_active = True
                    while input_active:
                        overlay = solved_overlay.copy()
                        # Draw input box
                        box_x = puzzle_ui.window.shape[1]//2-150
                        box_y = 380
                        cv2.rectangle(overlay, (box_x, box_y), (box_x+300, box_y+50), (180,255,80), 2)
                        cv2.putText(overlay, name+"_", (box_x+10, box_y+35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
                        # Draw arrow (simulate button)
                        cv2.putText(overlay, "→", (box_x+260, box_y+38), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180,255,80), 3)
                        # Play again
                        cv2.putText(overlay, "Click & Play Again", (puzzle_ui.window.shape[1]//2-120, box_y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,255,80), 2)
                        cv2.imshow("Puzzle", overlay)
                        key = cv2.waitKey(0)
                        if key == 13 or key == 10:  # Enter
                            input_active = False
                        elif key == 8:  # Backspace
                            name = name[:-1]
                        elif 32 <= key <= 126 and len(name) < 16:
                            name += chr(key)
                    # After name input, wait and break
                    cv2.putText(overlay, f"Thanks, {name}!", (puzzle_ui.window.shape[1]//2-120, box_y+140), cv2.FONT_HERSHEY_SIMPLEX, 1, (180,255,80), 2)
                    cv2.imshow("Puzzle", overlay)
                    cv2.waitKey(1500)
                    break
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

# Entry point to run the main function
if __name__ == "__main__":
    main()
