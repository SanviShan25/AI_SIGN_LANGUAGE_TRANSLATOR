# data_collection/collect_data.py

import cv2
import os
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================

# Sign class you want to record
SIGN_NAME = "hello"

# Number of videos to collect
NUM_VIDEOS = 100

# Duration of each video (seconds)
VIDEO_DURATION = 3

# Webcam FPS
FPS = 20

# =====================================================
# CREATE FOLDER
# =====================================================

save_dir = f"datasets/raw_videos/{SIGN_NAME}"
os.makedirs(save_dir, exist_ok=True)

# =====================================================
# OPEN WEBCAM
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

print(f"\nCollecting data for sign: {SIGN_NAME}")
print(f"Target videos: {NUM_VIDEOS}")

video_count = 0

while video_count < NUM_VIDEOS:

    print(f"\nPress 'R' to record video {video_count+1}")
    print("Press 'Q' to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Sign: {SIGN_NAME}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            display_frame,
            f"Recorded: {video_count}/{NUM_VIDEOS}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.imshow("Data Collection", display_frame)

        key = cv2.waitKey(1)

        # Start recording
        if key == ord('r'):
            break

        # Quit
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # =================================================
    # CREATE VIDEO FILE
    # =================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_path = os.path.join(
        save_dir,
        f"{SIGN_NAME}_{timestamp}.avi"
    )

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        video_path,
        fourcc,
        FPS,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )

    print("Recording...")

    total_frames = VIDEO_DURATION * FPS

    frame_counter = 0

    while frame_counter < total_frames:

        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            "RECORDING",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        cv2.imshow("Data Collection", display_frame)

        cv2.waitKey(1)

        frame_counter += 1

    out.release()

    video_count += 1

    print(f"Saved: {video_path}")

cap.release()
cv2.destroyAllWindows()

print("\nData Collection Completed")