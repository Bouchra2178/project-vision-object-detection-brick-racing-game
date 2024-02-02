import cv2
import math
import numpy as np
from app_work2 import get_limits
from clock import *

yellow = [0, 255, 255]
yellow = [255, 0, 0]  #bleuuuu
cap = cv2.VideoCapture(0)
previous_x_position = 0
while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = custom_in_range(hsvImage, lowerLimit, upperLimit)
    # nonzero_pixels = my_argwhere(mask)
    nonzero_pixels=np.argwhere(mask>0)
    if nonzero_pixels.size > 0:
        # Get the x-coordinates of non-zero pixels
        x_coordinates = nonzero_pixels[:, 1]
        # Calculate the mean x-coordinate
        x_position = np.mean(x_coordinates, dtype=int)
        y1, x1 = np.min(nonzero_pixels, axis=0)
        y2, x2 = np.max(nonzero_pixels, axis=0)
        x_position = (x1 + x2) // 2
        if previous_x_position!=None and abs(x_position-previous_x_position)>2 :
            print(x_position-previous_x_position)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        previous_x_position = x_position
        cv2.putText(frame, f'X Position: {x_position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
           previous_x_position=None
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

