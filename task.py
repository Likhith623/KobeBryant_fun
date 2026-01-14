import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# --- Setup Paths ---
base_img_path = '/Users/likhith./KobeBryant_fun/kobe_nba.png'
hair_png_path = '/Users/likhith./KobeBryant_fun/hair.png'
beard_png_path = '/Users/likhith./KobeBryant_fun/beard.png'

# --- Visualization Helper ---
def show_image(img, title, cmap=None, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    
    if len(img.shape) == 2:
        ax.imshow(img, cmap=cmap or 'gray')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    ax.set_title(title)
    ax.axis('off')

# ==========================================
# Task 1: Read, Verify & Metadata Annotation
# ==========================================
print("--- Task 1: IO & Annotation ---")
img_bgr = cv2.imread(base_img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {base_img_path}")

h, w, c = img_bgr.shape
img_annotated = img_bgr.copy()
label = f"Res: {w}x{h} | Mode: RGB"

cv2.rectangle(img_annotated, (10, h-40), (250, h-10), (0,0,0), -1)
cv2.putText(img_annotated, label, (20, h-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

show_image(img_annotated, "Task 1: Original with Metadata")
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task1_annotated.jpg"), img_annotated)

# ==========================================
# Task 2: Channel Splitting
# ==========================================
print("--- Task 2: Channel Splitting ---")
B, G, R = cv2.split(img_bgr)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
show_image(R, "Red Channel", cmap='Reds', ax=axs[0])
show_image(G, "Green Channel", cmap='Greens', ax=axs[1])
show_image(B, "Blue Channel", cmap='Blues', ax=axs[2])
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task2_red.jpg"), R)
cv2.imwrite(os.path.join(OUTPUT_DIR, "task2_green.jpg"), G)
cv2.imwrite(os.path.join(OUTPUT_DIR, "task2_blue.jpg"), B)

# ==========================================
# Task 3: Adaptive Thresholding
# ==========================================
print("--- Task 3: Thresholding ---")
gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

binary_adaptive = cv2.adaptiveThreshold(
    gray_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

show_image(binary_adaptive, "Task 3: Adaptive Binary Map")
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task3_adaptive_threshold.jpg"), binary_adaptive)

# ==========================================
# Task 4 & 5: Geometric Transforms
# ==========================================
print("--- Task 4 & 5: Geometry ---")
scale_percent = 50
width = int(w * scale_percent / 100)
height = int(h * scale_percent / 100)
resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)

M = cv2.getRotationMatrix2D((w//2, h//2), -15, 1)
rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_image(resized, "Task 4: Resized", ax=axs[0])
show_image(rotated, "Task 5: Rotated", ax=axs[1])
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task4_resized.jpg"), resized)
cv2.imwrite(os.path.join(OUTPUT_DIR, "task5_rotated.jpg"), rotated)

# ==========================================
# Task 6: Canny Edge Detection
# ==========================================
print("--- Task 6: Canny Edge Detection ---")
edges = cv2.Canny(img_bgr, 100, 200)

show_image(edges, "Task 6: Canny Edges", cmap='gray')
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task6_canny_edges.jpg"), edges)

# ==========================================
# Task 7: Gaussian Blur
# ==========================================
print("--- Task 7: Smoothing/Blurring ---")
blurred = cv2.GaussianBlur(img_bgr, (15, 15), 0)

show_image(blurred, "Task 7: Gaussian Blur")
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task7_gaussian_blur.jpg"), blurred)

# ==========================================
# Advanced Task: MediaPipe Face Filter
# ==========================================
print("--- Advanced Task: Precision Face Overlay ---")

frame = img_bgr.copy()
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

sunglasses_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread("moustache.png", cv2.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

results = face_mesh.process(rgb)
if not results.multi_face_landmarks:
    raise RuntimeError("❌ No face detected")

face_landmarks = results.multi_face_landmarks[0]

def get_point(i):
    lm = face_landmarks.landmark[i]
    return int(lm.x * w), int(lm.y * h)

def overlay(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    alpha = fg[:,:,3] / 255.0
    for c in range(3):
        bg[y:y+h_fg, x:x+w_fg, c] = (
            alpha * fg[:,:,c] +
            (1 - alpha) * bg[y:y+h_fg, x:x+w_fg, c]
        )
    return bg

# Sunglasses
left_eye_outer = get_point(33)
right_eye_outer = get_point(263)
left_eye_bottom = get_point(145)
right_eye_bottom = get_point(374)

eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
eye_center_y = (left_eye_bottom[1] + right_eye_bottom[1]) // 2

eye_width = int(np.linalg.norm(
    np.array(left_eye_outer) - np.array(right_eye_outer)
))

sg_width = int(eye_width * 2.0)
sg_height = int(sg_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])
sunglasses_resized = cv2.resize(sunglasses_img, (sg_width, sg_height))

sg_x = eye_center_x - sg_width // 2
sg_y = eye_center_y - sg_height // 2 - 20

frame = overlay(frame, sunglasses_resized, sg_x, sg_y)

# Mustache
nose_tip = get_point(4)
upper_lip = get_point(13)
left_temple = get_point(71)
right_temple = get_point(301)

face_width = int(np.linalg.norm(
    np.array(left_temple) - np.array(right_temple)
))

must_width = int(face_width * 0.9)
must_height = int(must_width * mustache_img.shape[0] / mustache_img.shape[1])
mustache_resized = cv2.resize(mustache_img, (must_width, must_height))

must_x = nose_tip[0] - must_width // 2 - 30
must_y = (nose_tip[1] + upper_lip[1]) // 2 - must_height // 2 + 10

frame = overlay(frame, mustache_resized, must_x, must_y)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Task 8: MediaPipe Filters")
plt.axis("off")
plt.show()

cv2.imwrite(os.path.join(OUTPUT_DIR, "task8_mediapipe_filters.jpg"), frame)

print("✅ All task outputs saved in /output folder")
