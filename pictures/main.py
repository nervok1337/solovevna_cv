import cv2
import numpy as np
from skimage.measure import label, regionprops

cap = cv2.VideoCapture('output.avi')

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

cnt = 0

while True:
    ret, frame = cap.read()

    if not ret:
            break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary1 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    labeled1 = label(binary1)
    _, binary2 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    labeled2 = label(binary2)

    regions = regionprops(labeled2)

    total_holes = sum(1 - region.euler_number for region in regions)

    if np.max(labeled1) == 4 and np.max(labeled2) == 2 and total_holes == 1:
        cnt+=1

print(cnt)


