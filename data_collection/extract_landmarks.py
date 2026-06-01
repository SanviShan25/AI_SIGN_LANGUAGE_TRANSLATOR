import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn import
# =====================================================
# CONFIGURATION
# =====================================================

DATASET_ROOT = "datasets/WLASL/dataset/SL"
OUTPUT_ROOT = "datasets/landmarks"

# Start with only 3 classes for testing
CLASSES = [
    "write",
    "yes",
    "you"
]

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# MEDIAPIPE HANDS
# =====================================================

mp_hands = mp.solutions.hands

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print(f"\nFound {len(CLASSES)} classes")
print("=" * 50)

# =====================================================
# PROCESS EACH CLASS
# =====================================================

for class_name in CLASSES:

    class_folder = os.path.join(
        DATASET_ROOT,
        class_name
    )

    if not os.path.exists(class_folder):
        print(f"\nSkipping: {class_name} (folder not found)")
        continue

    save_folder = os.path.join(
        OUTPUT_ROOT,
        class_name
    )

    os.makedirs(save_folder, exist_ok=True)

    video_files = [
        f for f in os.listdir(class_folder)
        if f.endswith(".mp4")
    ]

    print(
        f"\nProcessing '{class_name}' "
        f"({len(video_files)} videos)"
    )

    for video_file in tqdm(video_files):

        video_path = os.path.join(
            class_folder,
            video_file
        )

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hand_detector.process(rgb)

            landmarks = []

            if results.multi_hand_landmarks:

                for hand in results.multi_hand_landmarks:

                    for lm in hand.landmark:

                        landmarks.extend([
                            lm.x,
                            lm.y,
                            lm.z
                        ])

            # =================================================
            # FORCE FIXED SIZE
            # 2 hands × 21 landmarks × 3 coords = 126
            # =================================================

            while len(landmarks) < 126:
                landmarks.append(0)

            landmarks = landmarks[:126]

            sequence.append(landmarks)

        cap.release()

        # Skip empty videos
        if len(sequence) == 0:
            continue

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        save_path = os.path.join(
            save_folder,
            video_file.replace(
                ".mp4",
                ".npy"
            )
        )

        np.save(
            save_path,
            sequence
        )

print("\nLandmark Extraction Complete!")
