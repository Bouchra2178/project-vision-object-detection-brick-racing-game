import cv2
import numpy as np
import keyboard
import random
import streamlit as st

def custom_in_range(frame, down, up):
    #height, width = len(frame), len(frame[0])
    height, width = 250,100 
    mask=np.zeros((frame.shape[0],frame.shape[1]),frame.dtype)

    for i in range(150):
        for j in range(150):
            if down[0] <= frame[i+height][j+width][0] <= up[0] and down[1] <= frame[i+height][j+width][1] <= up[1] and down[2] <= frame[i+height][j+width][2] <= up[2]:
                mask[i+height][j+width] = 255

    return mask
cars = ['pickup_truck.png', 'semi_trailer.png',"car2.png","car.png", 'taxi.png', 'van.png',"enemy_car_1.png","enemy_car_2.png"]
choice=0
background_height = 500
background_width = 500
gray = (100, 100, 100)
black=(0,0,0)
green = (76, 208, 56)
red = (200, 0, 0)
white = (255, 255, 255)
yellow = (255, 232, 0)
road_width = 300
marker_width = 10
marker_height = 50
left_lane = 150
center_lane = 250
right_lane = 350
road = (100, 0, road_width, 500)
left_edge_marker = (95, 0, marker_width, 500)
right_edge_marker = (395, 0, marker_width, 500)
move = 0
speed = 2
score = 0
def ROAD(frame):
    global move 
    frame = np.ones((500, 500, 3), dtype=np.uint8) * red
    cv2.rectangle(frame, (100, 0), (100 + road_width, 500), black, -1)
    cv2.rectangle(frame, (95, 0), (95 + marker_width, 500), yellow, -1)
    cv2.rectangle(frame, (395, 0), (395 + marker_width, 500), yellow, -1)
    move += speed * 4
    if move >= marker_height * 2:
        move = 0
    for y in range(marker_height * -2, 500, marker_height * 2):
        cv2.rectangle(frame, (left_lane + 45, y + move), (left_lane + 45 + marker_width, y + move + marker_height), white, -1)
        cv2.rectangle(frame, (center_lane + 45, y + move), (center_lane + 45 + marker_width, y + move + marker_height), white, -1)
    frame = frame.astype(np.uint8)
    return frame
def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit
def add_obstacle(img, occupied_lines):
    height, width = img.shape[:2]
    # Generate random x-coordinate for the obstacle
    x_coord = random.randint(100, width-200)
    obstacle_height = 50
    y_coord = random.randint(0, height - obstacle_height)
    while any(y_coord <= line <= y_coord + obstacle_height for line in occupied_lines):
        y_coord = random.randint(0, height - obstacle_height)

    return [x_coord, 0, x_coord + 50, 50]
def add_obstacle2(img):
    height, width = img.shape[:2]
    global cars
    global choice
    choice = random.randint(0, len(cars)-1)
    obstacle_img = cv2.imread("./cars/"+cars[choice])
    obstacle_height, obstacle_width = obstacle_img.shape[:2]
    x_coord = random.randint(95+obstacle_width, 395-obstacle_width)
    y_coord = 0 
    img[y_coord:y_coord + obstacle_height, x_coord:x_coord + obstacle_width] = obstacle_img
    return img, (x_coord, y_coord)
def move_obstacle(img, obstacle_position, move_distance=10):
    global choice
    global cars
    global score
    #move_distance=30
    move_distance=min(int(10+ 0.8*(score+1)),30)
    x, y = obstacle_position
    new_y = y + move_distance
    obstacle_img = cv2.imread("./cars/"+cars[choice])
    obstacle_height, obstacle_width = obstacle_img.shape[:2]
    img[y:y + obstacle_height, x:x + obstacle_width] = 0 
    if (new_y + obstacle_height>500):
        img, (x, new_y)=add_obstacle2(img)
        score+=1
        return img, (x, new_y),score,move_distance
        # img[new_y:400, x:x + obstacle_width] = obstacle_img[400-new_y,:]
    else:
        img[new_y:new_y + obstacle_height, x:x + obstacle_width] = obstacle_img 

    return img, (x, new_y),score,move_distance
def car(background,x):
    car_image = cv2.imread("./cars/enemy_car_2.png")
    car_height, car_width = car_image.shape[:2]
    x_center = x
    y_bottom = background.shape[0]
    x1 = x_center - car_width // 2
    y1 = y_bottom - (car_height+20)
    x2 = x_center + car_width // 2
    y2 = y_bottom -20
    background[y1:y2, x1:x2] = car_image
    return x1, x2, y1
def load_background_image(background_height, background_width, image_path):
    #fonction pour definir bachground avec une image
    background = cv2.imread(image_path)

    background = cv2.resize(background, (background_width, background_height))

    return background

