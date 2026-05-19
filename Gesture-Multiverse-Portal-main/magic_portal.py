import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import face_recognition
import math
import time

# -------------------- LOAD FACE LANDMARKER --------------------
model_path = "face_landmarker.task"

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# -------------------- LOAD AUTHORIZED FACE --------------------
authorized_image = face_recognition.load_image_file("authorized_face.jpg")
authorized_encoding = face_recognition.face_encodings(authorized_image)[0]

# -------------------- LOAD WORLDS --------------------
world_images = [
    cv2.imread("worlds/space.jpg"),
    cv2.imread("worlds/forest.jpeg"),
    cv2.imread("worlds/ocean.jpg"),
    cv2.imread("worlds/fire.jpg"),
    cv2.imread("worlds/city.jpg")
]

current_world = 0
portal_open = False
authorized = False

blink_counter = 0
blink_cooldown = 0
angle_offset = 0

# -------------------- BLINK DETECTION --------------------
def is_blinking(blendshapes):
    blink_left = 0
    blink_right = 0

    for shape in blendshapes:
        if shape.category_name == "eyeBlinkLeft":
            blink_left = shape.score
        if shape.category_name == "eyeBlinkRight":
            blink_right = shape.score

    return blink_left > 0.5 and blink_right > 0.5

# -------------------- FACE RECOGNITION --------------------
def check_authorization(frame):
    global authorized
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_encodings(rgb)

    if len(faces) > 0:
        match = face_recognition.compare_faces(
            [authorized_encoding], faces[0], tolerance=0.5
        )
        authorized = match[0]
    else:
        authorized = False

# -------------------- EYE GLOW EFFECT --------------------
def draw_eye_glow(frame, landmarks):
    h, w, _ = frame.shape

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    for eye in [left_eye, right_eye]:
        x = int(eye.x * w)
        y = int(eye.y * h)

        for r in range(10, 40, 10):
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), r, (255, 255, 0), -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

# -------------------- MAGIC FACE AURA --------------------
def draw_face_aura(frame, landmarks, angle):
    h, w, _ = frame.shape

    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]

    center = (int(np.mean(xs)), int(np.mean(ys)))
    radius = int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.8)

    for i in range(3):
        overlay = frame.copy()
        r = radius + i * 20
        cv2.circle(overlay, center, r, (255, 0, 255), 3)
        cv2.addWeighted(overlay, 0.2 - i*0.05, frame, 1 - (0.2 - i*0.05), 0, frame)

# -------------------- PORTAL OVERLAY --------------------
def overlay_world(frame, world_img, center, radius):
    world_resized = cv2.resize(world_img, (2*radius, 2*radius))
    mask = np.zeros((2*radius, 2*radius), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    portal_area = frame[
        center[1]-radius:center[1]+radius,
        center[0]-radius:center[0]+radius
    ]

    if portal_area.shape[:2] != world_resized.shape[:2]:
        return frame

    world_masked = cv2.bitwise_and(world_resized, world_resized, mask=mask)
    bg_masked = cv2.bitwise_and(portal_area, portal_area, mask=cv2.bitwise_not(mask))

    frame[
        center[1]-radius:center[1]+radius,
        center[0]-radius:center[0]+radius
    ] = cv2.add(world_masked, bg_masked)

    return frame

# -------------------- FIRE RING --------------------
def draw_fire_ring(frame, center, radius, angle_offset):
    sparks = 80
    for i in range(sparks):
        angle = (2 * math.pi / sparks) * i + angle_offset
        r = radius + np.random.randint(-8, 8)
        x = int(center[0] + r * math.cos(angle))
        y = int(center[1] + r * math.sin(angle))
        color = (0, np.random.randint(120, 200), 255)
        size = np.random.randint(2, 5)
        cv2.circle(frame, (x, y), size, color, -1)
    return frame

# -------------------- WEBCAM --------------------
cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_id += 1

    check_authorization(frame)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = landmarker.detect_for_video(mp_image, frame_id * 33)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0]

        if authorized:
            cv2.putText(frame, "AUTHORIZED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            draw_eye_glow(frame, landmarks)
            draw_face_aura(frame, landmarks, angle_offset)

            if blink_cooldown > 0:
                blink_cooldown -= 1

            if is_blinking(blendshapes) and blink_cooldown == 0:
                blink_counter += 1
                blink_cooldown = 15

            if not portal_open and blink_counter >= 1:
                portal_open = True
                blink_counter = 0

            elif portal_open and blink_counter >= 2:
                current_world = (current_world + 1) % len(world_images)
                blink_counter = 0

        else:
            cv2.putText(frame, "ACCESS DENIED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            portal_open = False

    else:
        portal_open = False

    if portal_open:
        h, w, _ = frame.shape
        center = (w//2, h//2)
        radius = 120
        frame = overlay_world(frame, world_images[current_world], center, radius)
        angle_offset += 0.15
        frame = draw_fire_ring(frame, center, radius + 5, angle_offset)

    cv2.imshow("Face Multiverse Portal", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
