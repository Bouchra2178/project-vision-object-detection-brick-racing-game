import streamlit as st
import cv2
import numpy as np
import keyboard
import random
from PIL import Image
from app_work2 import *
from filtres import *
from clock import *

def GAME(place):
    global score
    image_path = "./back_3.png"
    #background = load_background_image(background_height, background_width, image_path)
    #background = np.zeros((background_height, background_width, 3), dtype=np.uint8)
    background=0
    background=ROAD(background)
    choice = random.randint(0, len(cars)-1) #choix initiale de la voiture
    car1 = cv2.imread("./cars/"+cars[choice])
    obstacle_height, obstacle_width = car1.shape[:2]
    obstacles = []
    background,pos = add_obstacle2(background)
    obstacles.append(background)
    yellow = [255, 0, 0] 
    cap = cv2.VideoCapture(0)
    X=background.shape[1] // 2
    previous_x_position=None #Recuperer la position precedente pour mesurer le deplacement
    while True:
        ret, frame = cap.read()
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #transformation des couleur en HSV 
        lowerLimit, upperLimit = get_limits(color=yellow)  #fonction pour recuper les variable HSV 
        #mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        # background.fill(0)
        # background = load_background_image(background_height, background_width, image_path)
        mask = custom_in_range(hsvImage, lowerLimit, upperLimit)
        background=ROAD(background)
        # nonzero_pixels=np.argwhere(mask>0)
        nonzero_pixels=my_argwhere(mask)
        if nonzero_pixels.size > 0:
            x_coordinates = nonzero_pixels[:, 1]
            x_position = np.mean(x_coordinates, dtype=int)
            y1, x1 = np.min(nonzero_pixels, axis=0)
            y2, x2 = np.max(nonzero_pixels, axis=0)
            x_position = (x1 + x2) // 2
            if previous_x_position!=None and abs(x_position-previous_x_position)>2 :
                if (previous_x_position-x_position)>0:
                    X+=(previous_x_position-x_position+5)
                else:
                    X+=(previous_x_position-x_position-5)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            previous_x_position = x_position     
        x1,x2,y1=car(background,X) 
        #verifier les colision avec les autre obstacle
        if y1 < pos[1] + obstacle_height:
                    if x1 > pos[0] and x1 < pos[0] + obstacle_width or x1 + x2-x1 > pos[0] and x1 + x2-x1 <pos[0] + obstacle_width:
                        print("Game Over !!!")
                        score=0
                        break
        #verifier les colision avec les Bordures
        if x1<=95 or x2>395:
            print("Game Over !!!")
            score=0
            break
        for obstacle in obstacles:
            background,pos,score,vitesse=move_obstacle(background,pos) #Deplacer les obstacles
        cv2.putText(background, f"Score: {score}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
        cv2.putText(background, f"Speed: {vitesse}", (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
        place.image(background,channels="BGR")
        #Option de deplacer la voiture avec les touches
        if keyboard.is_pressed('left'):
            X -= 10
        elif keyboard.is_pressed('right'):
            X += 10
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    return 0

def part1(nav_option):
    st.header("Part 1 vision")
    if nav_option == "FILTRES":
        st.title("Image Filters App")

    # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

            # Choose filter type
            filter_type = st.selectbox("Choose a filter:", ['Ouverture', 'Fermeture', 'Filtre Médiane', 'Laplacien','Moyenne','erosion','dilatation','Gaussian','Sobel'])

            # Choose kernel size
            kernel_size = st.slider("Select kernel size:", min_value=3, max_value=15, step=2, value=5)
            filtered_img = apply_filter(img, filter_type, kernel_size)
            # Apply the selected filter
            img_normalized = normalize_image(img)
            filtered_img_normalized = normalize_image(filtered_img)

            # Display the original and filtered images
            st.image(img_normalized,channels="BGR", caption="Original Image", use_column_width=True)
            st.image(filtered_img_normalized,caption="Filtered Image", use_column_width=True)

    elif nav_option == "Green screen":
        st.write('L\'effet de "Green screen", également connu sous le nom de chroma key, est une technique largement utilisée dans l\'industrie cinématographique et télévisuelle. Cette méthode consiste à remplacer une couleur spécifique, souvent le vert, par une autre image ou vidéo. Souvent utilisé pour créer des effets spéciaux et des arrière-plans virtuels, le "Green screen" permet aux réalisateurs de superposer des acteurs sur des environnements numériques. Dans le processus de production vidéo, la couleur verte est choisie car elle est rarement présente dans la plupart des tenues et des décors. L\'utilisation de cette technique nécessite une post-production pour combiner les images capturées avec l\'arrière-plan choisi, créant ainsi des scènes impossibles ou fantastiques.')
        image_button_clicked = st.button("Image Processing")
        uploaded_file1 = st.file_uploader("Choose an image 1...", type=["jpg", "jpeg", "png"], key="image1")
        uploaded_file2 = st.file_uploader("Choose un background", type=["jpg", "jpeg", "png"], key="image2")

        if image_button_clicked:  
            frame = None
            image = None
            # Check if the first image is uploaded
            if uploaded_file1 is not None:
                # Read the uploaded image
                frame = cv2.imdecode(np.fromstring(uploaded_file1.read(), np.uint8), 1)

            # Check if the second image is uploaded
            if uploaded_file2 is not None:
                # Read the uploaded image
                image = cv2.imdecode(np.fromstring(uploaded_file2.read(), np.uint8), 1)
            # Process the images if both are uploaded
            if frame is not None and image is not None:
                green_screen_processing(frame, image)

                    
        video_button_clicked = st.button("Video Processing")
        video_file = st.file_uploader("Choose a video...", type=["mp4"])
        if video_button_clicked:
                if video_file is  None:
                    # Appeler votre fonction de traitement vidéo
                    st.warning("Please upload an image before processing.")
                else:
                    greenScreenVideo()
            

    elif nav_option == "Invisibilitie Clock":
        st.write('La cape d\'invisibilité, souvent associée à la fiction et à la magie, est un concept de longue date qui a captivé l\'imagination à travers des récits fantastiques. Popularisée par des œuvres telles que la série "Harry Potter", la cape d\'invisibilité est un artefact fictif qui confère à son porteur la capacité de devenir invisible. Dans le monde réel, des tentatives ont été faites pour créer des effets d\'invisibilité en utilisant des techniques de vision par ordinateur. Ces méthodes, bien que ne conférant pas une invisibilité totale, illustrent l\'application créative des technologies modernes pour réaliser des idées autrefois considérées comme purement imaginaires.')

        st.write("Vous avez choisi Invisibilitie gaust")
        invisibility_gaust()
        
    elif nav_option=="Detection objet Couleur":
        st.header("Detection Objet Par couleur")
        if st.button("Detection"):
            detetion_objet_couleur()
     
def part2():
    st.header("Part2 Project vision")
    place=st.empty()
    if st.button("Start"):
        GAME(place)
    
def main():
    nav_option = st.sidebar.selectbox("Part", ["PART1", "PART2"])
    if nav_option == "-":
        st.title("Home Page")
        st.header("welcom to vision gui")
    elif nav_option == "PART1":
        nav_option = st.sidebar.radio("Navigation", ["FILTRES","Detection objet Couleur", "Green screen", "Invisibilitie Clock"])
        part1(nav_option)
    elif nav_option == "PART2":
        part2()

if __name__ == '__main__':
    main()