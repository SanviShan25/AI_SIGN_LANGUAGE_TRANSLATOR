import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2
)

print("MediaPipe Hands Loaded Successfully")