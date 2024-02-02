import streamlit as st
import cv2
import numpy as np
from filtres import *
from app_work2 import *
#rgb(9,137,35) voiture noir
#rgb(109,254,1) youtube

def my_argwhere(mask):
    indices = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                indices.append((i, j))
    return np.array(indices)

def where(frame,image,mask):
    height, width = len(frame), len(frame[0])
    for i in range(height):
        for j in range(width):
            if mask[i,j]==0:
                frame[i,j,:]=image[i,j,:]
    
    return frame



    
    return frame               
def resize_image(image, new_size):
    height, width = image.shape[:2]
    new_width,new_height = new_size

    # Calculer les ratios de redimensionnement
    height_ratio = height / new_height
    width_ratio = width / new_width

    # Initialiser la nouvelle image avec la taille désirée
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Coordonnées originales dans l'image
            original_i = i * height_ratio
            original_j = j * width_ratio

            # Coordonnées entières supérieures et inférieures
            i_low, j_low = int(original_i), int(original_j)
            i_high, j_high = min(i_low + 1, height - 1), min(j_low + 1, width - 1)

            # Interpolation bilinéaire
            weight_i, weight_j = original_i - i_low, original_j - j_low
            top_left = image[i_low, j_low] * (1 - weight_i) * (1 - weight_j)
            top_right = image[i_low, j_high] * (1 - weight_i) * weight_j
            bottom_left = image[i_high, j_low] * weight_i * (1 - weight_j)
            bottom_right = image[i_high, j_high] * weight_i * weight_j

            # Valeur interpolée
            resized_image[i, j] = top_left + top_right + bottom_left + bottom_right

    return resized_image

def custom_in_range2(frame, down, up):
    height, width = len(frame), len(frame[0])
    mask=np.zeros((frame.shape[0],frame.shape[1]),frame.dtype)

    for i in range(height):
        for j in range(width):
            if down[0] <= frame[i][j][0] <= up[0] and down[1] <= frame[i][j][1] <= up[1] and down[2] <= frame[i][j][2] <= up[2]:
                mask[i][j] = 255


    return mask
def custom_in_range(frame, down, up):
    #height, width = len(frame), len(frame[0])
    height, width = 100,100
    mask=np.zeros((frame.shape[0],frame.shape[1]),frame.dtype)
    # mask[0:height,:] =255
    # mask[480-height:,:] =255
    # mask[:,0:width] =255
    for i in range(300):
        for j in range(400):
            if down[0] <= frame[i+height][j+width][0] <= up[0] and down[1] <= frame[i+height][j+width][1] <= up[1] and down[2] <= frame[i+height][j+width][2] <= up[2]:
                mask[i+height][j+width] = 255


    return mask

# Définir les fonctions nécessaires
def custom_bitwise_and(image, mask):
    result = np.zeros_like(image)
    result[mask != 0] = image[mask != 0]
    return result

def custom_bitwise_not2(mask):
    return 255 - mask

def green_screen_processing(frame, image):
    # Resize the frame and image
    frame = resize_image(frame, (640, 480))
    image = resize_image(image, (640, 480))
    #on doit dabort dtranspormeer les couleur en format rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    u_green = np.array([70, 153, 104])
    l_green = np.array([0, 30, 30])
    # Create a mask to detect only green
    mask = custom_in_range2(frame, l_green, u_green)
    mask_free = custom_bitwise_not2(mask)
    # mask = custom_bitwise_not2(mask)
    # Apply the mask to get green pixels from the image; all other pixels are set to black
    res = custom_bitwise_and(frame, mask=mask)
    # Extract other pixels from the frame
    f = frame - res
    # Replace black pixels with the chosen background
    # f = np.where(f == 0, image, f)
    f=where(f,image,mask_free)
    # Display images in the Streamlit interface
    st.image([frame, f], caption=['Original', 'Resultat'], channels="RGB",use_column_width=True)
    st.image(mask, caption='MASK',use_column_width=True)

def capture_background():
    cap = cv2.VideoCapture(0)

    st.write("Capturing Background:")
    captured_image = st.image([])

    button_stop_capture = st.button("Stop Capture")

    while cap.isOpened() and not button_stop_capture:
        ret, background = cap.read()
        if ret:
            # Display the captured background in the Streamlit app
            captured_image.image(background, channels="BGR", use_column_width=True)

            # Save the background image
            cv2.imwrite("image.jpg", background)

            if cv2.waitKey(5) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def invisibility_gaust():
    # Capture the background image
    capture_background()

    # Load the captured background image
    background = cv2.imread('./image.jpg')

    # Open the camera for the invisibility effect
    cap = cv2.VideoCapture(0)

    st.write("Fenêtre de la caméra:")
    camera_image = st.image([])

    button_stop_camera = st.button("Arrêter la caméra")

    while cap.isOpened() and not button_stop_camera:
        ret, current_frame = cap.read()
        if ret:
            hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

            l_red = np.array([0, 120, 170])
            u_red = np.array([10, 255, 255])
            mask1 = custom_in_range(hsv_frame, l_red, u_red)

            l_red = np.array([170, 120, 70])
            u_red = np.array([180, 255, 255])
            mask2 = custom_in_range(hsv_frame, l_red, u_red)

            red_mask = mask1 + mask2
            """
            for i in range(10):
             red_mask = ouverture_filtre2(red_mask)

            red_mask = dilatation2(red_mask)

            #red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10)
            #red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
            """
            part1 = custom_bitwise_and(background, mask=red_mask)
            part2 = custom_bitwise_and(current_frame, mask=custom_bitwise_not2(red_mask))

            result = part1 + part2

            # Display the camera image in the interface
            camera_image.image(result, channels="BGR", use_column_width=True)

    cap.release()

def greenScreenVideo():
        
    video  = cv2.VideoCapture("./green_screen/green.mp4")
    image = cv2.imread("./green_screen/gbg.jpeg")
    place2=st.empty()
    u_green = np.array([104, 153, 70])
    l_green = np.array([30, 30, 0])
    Stop_video = st.button("Stop Video")
    while not Stop_video:
        ret, frame = video.read()
        frame = cv2.resize(frame, (640, 480))
        image = cv2.resize(image, (640, 480))
        #creer un mask pour detecter que les pixel de la couleur vert
        mask = custom_in_range2(frame, l_green, u_green)
        mask_free = custom_bitwise_not2(mask)
        #dans res recuperer que les pixel vert de image tout autre pixel sera mit a noir 
        res = custom_bitwise_and(frame, mask=mask)
        #recuper les autre pixel de image dans f
        f = frame - res
        #puis Si les pixel sont a null on les remplace par le bachground chosit Sinn on fait rien
        f=where(f,image,mask_free)
        # f = np.where(f==0, image, f)
        place2.image(f,channels="BGR")
        if cv2.waitKey(1) == 27:
            break 

    video.release()
    cv2.destroyAllWindows()

def my_argwhere(mask):
    indices = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                indices.append((i, j))
    return np.array(indices)

def detetion_objet_couleur():
    yellow = [0, 255, 255]
    yellow = [255, 0, 0]#bleuuuu
    cap = cv2.VideoCapture(0)
    previous_x_position = 0
    place3=st.empty()
    Stop_detection=st.button("Stop Detection")
    while not Stop_detection:
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
            print(x_position)
            if previous_x_position!=None and abs(x_position-previous_x_position)>2 :
                print(x_position-previous_x_position)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            previous_x_position = x_position
            cv2.putText(frame, f'X Position: {x_position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            previous_x_position=None
            
        place3.image(frame,"Detection Object",channels="BGR")
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break