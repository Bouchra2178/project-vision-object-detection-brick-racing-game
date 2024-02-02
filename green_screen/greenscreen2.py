import cv2 
import numpy as np 

def custom_in_range(image, lower_bound, upper_bound):
    # Créer un masque manuellement
    mask = np.zeros_like(image)

    # Boucler à travers chaque pixel de l'image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Utiliser np.all() pour vérifier la condition pour chaque canal de couleur
            if np.all(lower_bound <= image[i, j]) and np.all(image[i, j] <= upper_bound):
                mask[i, j] = 255  # Mettre à blanc (255) si la condition est vraie

    return mask

def SIIRange(hsvImage, lower_bound, upper_bound):
    
    # Créer un masque manuellement
    mask = np.zeros_like(hsvImage)

    # Boucler à travers chaque pixel de l'image
    for i in range(hsvImage.shape[0]):
        for j in range(hsvImage.shape[1]):
            # Utiliser np.all() pour vérifier la condition pour chaque canal de couleur
            if np.all(lower_bound <= hsvImage[i, j]) and np.all(hsvImage[i, j] <= upper_bound):
                mask[i, j] = hsvImage[i, j]

    return mask
def custom_bitwise_and(image, mask):
    # Create an empty image with the same shape as the input image
    result = np.zeros_like(image)
    # Copy pixels from the input image where the mask is not zero
    result[mask != 0] = image[mask != 0]
    return result
    
def custom_bitwise_not2(mask):
    return 255 - mask
# video  = cv2.VideoCapture("./green_screen/green.mp4")
frame=cv2.imread("./green_screen/grreen2.jpg")
image = cv2.imread("./green_screen/gbg.jpeg")

frame = cv2.resize(frame, (640, 480))
image = cv2.resize(image, (640, 480))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

u_green = np.array([104, 153, 70])
l_green = np.array([30,30,0])
#creer un mask pour detecter que les pixel de la couleur vert
mask = custom_in_range(frame, l_green, u_green)
#mask = SIIRange(frame, l_green, u_green)

mask = custom_bitwise_not2(mask)
#dans res recuperer que les pixel vert de image tout autre pixel sera mit a noir 
res = custom_bitwise_and( frame, mask=mask)
#recuper les autre pixel de image dans f
f = frame - res
#puis Si les pixel sont a null on les remplace par le bachground chosit Sinn on fait rien
f = np.where(f==0, image, f)
cv2.imshow("video", frame)
cv2.imshow("mask", f)
cv2.imshow("mask2", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

