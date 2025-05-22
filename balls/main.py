import cv2
import numpy as np
import os
import json
import random

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -7)

def sort_balls(curr_balls, balls_per_row=2, y_threshold=10):
    balls = list(curr_balls.items())
    balls.sort(key=lambda item: (item[1][1], item[1][0]))

    rows = []
    current_row = []
    last_y = None

    for ball in balls:
        _, (_, y) = ball
        if not current_row:
            current_row.append(ball)
            last_y = y
        elif abs(y - last_y) <= y_threshold and len(current_row) < balls_per_row:
            current_row.append(ball)
        else:
            rows.append(sorted(current_row, key=lambda item: item[1][0]))
            current_row = [ball]
            last_y = y

    if current_row:
        rows.append(sorted(current_row, key=lambda item: item[1][0]))

    sorted_balls = [ball for row in rows for ball in row]
    return dict(sorted_balls)




def get_color(image):
    x,y,w,h = cv2.selectROI("Color selection", image)
    x,y,w,h = int(x),int(y),int(w),int(h)
    roi = image[y:y+h, x:x+w]

    color = (np.median(roi[:,:,0]),
             np.median(roi[:,:,1]),
             np.median(roi[:,:,2]))
    
    cv2.destroyWindow("Color selection")
    return color

def get_ball(image, color):
    lower = (np.max([0, color[0] - 5]), color[1] * 0.8, color[2] * 0.8)
    upper = (color[0] + 5, 255, 255)
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return True, (int(x), int(y), int(radius), mask)
    return False, (-1, -1, -1, np.array([]))


cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
file_name = "settings.json"
if os.path.exists(file_name):
    base_colors = json.load(open(file_name, "r"))
else:
    base_colors = {}

game_started = False
guess_colors = []
curr_balls = {}
while capture.isOpened():
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame,(7,7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    key = chr(cv2.waitKey(1) & 0xFF)
    if key in "1234":
        color = get_color(hsv)
        if np.isnan(color).any():
            pass
        else:
            base_colors[f"{key}"] = color
            print(base_colors)
    if key == "q":
        break
    for key in base_colors:
        retr, (x, y, radius, mask) = get_ball(hsv, base_colors[key])
        if retr:
            cv2.imshow("Mask", mask)
            cv2.circle(frame, (x,y), radius, (255, 0, 255), 2)
        curr_balls[key] = (x, y)
        if len(base_colors) == 3:
            curr_colors = list(sort_balls(curr_balls, 3).keys())
        elif len(base_colors) == 4:
            curr_colors = list(sort_balls(curr_balls, 2).keys())

    if len(base_colors) >= 3:
        if not game_started:
            guess_colors = list(base_colors)
            random.shuffle(guess_colors)
            print(guess_colors)
            game_started = True
        if game_started and (curr_colors == guess_colors) and (len(curr_colors) == len(guess_colors)):
            cv2.putText(frame, f"YOU WON!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        

    cv2.putText(frame, f"Game started with {len(base_colors)} colors = {game_started}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()
json.dump(base_colors, open(file_name, "w"))
