import numpy as np
import cv2
import streamlit as st

def normalize_image(img):
    # Normalize pixel values to be in the range [0.0, 1.0]
    img_min, img_max = img.min(), img.max()
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

def apply_filter(img, filter_type, kernel_size):
    if filter_type == 'Ouverture':
        return ouverture_filtre(img, kernel_size)
    elif filter_type == 'Fermeture':
        return fermeture_filtre(img, kernel_size)
    elif filter_type == 'Moyenne':
        return mean_filter(img,kernel_size)
    elif filter_type == 'Filtre Médiane':
        return filtreMediane(img, kernel_size)
    elif filter_type == 'Laplacien':
        return laplacian_filter(img)
    elif filter_type == 'erosion':
        return erosion(img,kernel_size)
    elif filter_type == 'dilatation':
        return dilatation(img, kernel_size)
    elif filter_type == 'Gaussian':
        return apply_gaussian_filter(img,kernel_size)
    elif filter_type == 'Sobel':
        return sobel_operator(img,kernel_size)
    else:
        return img
    

def sobel_operator(image,kernel_size):
    # Convertir l'image en niveaux de gris si elle est en couleur
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Définir les noyaux Sobel
        #kernel = np.ones((kernel_size, kernel_size), np.uint8)

    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Appliquer les opérations de convolution
    sobel_x = convolution(gray_image, sobel_x_kernel)
    sobel_y = convolution(gray_image, sobel_y_kernel)

    # Calculer la magnitude du gradient
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normaliser la magnitude du gradient
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    return gradient_magnitude.astype(np.uint8)

def convolution(image, kernel):
    # Taille de l'image et du noyau
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Taille de la sortie
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialiser la sortie avec des zéros
    output = np.zeros((output_height, output_width))

    # Appliquer l'opération de convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def mean_filter(img, vois):
   
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    imgMean = np.zeros(img.shape, img.dtype)

    for y in range(h):
        for x in range(w):
            if y < vois // 2 or y > h - vois // 2 or x < vois // 2 or x > w - vois // 2:
                imgMean[y, x] = img[y, x]
            else:
                imgVois = img[y - vois // 2:y + vois // 2 + 1, x - vois // 2:x + vois // 2 + 1]
                imgMean[y, x] = np.mean(imgVois)

    return imgMean
    
def apply_gaussian_filter(image, kernel_size=5):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img2=cv2.GaussianBlur(image, (3, 3), 0)
    #st.image(img2, caption=f"Eroded Image (Kernel Size: {kernel_size})", use_column_width=True)

    def create_gaussian_kernel(kernel_size, sigma=1.0):
        kernel_range = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1, 1)
        kernel = np.exp(-(kernel_range ** 2) / (2.0 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel.reshape((1, -1))

    def convolution(image, kernel):
        return cv2.filter2D(image, -1, kernel)

    # Convertir l'image en niveaux de gris si elle est en couleur
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer le filtre gaussien à partir de zéro
    kernel = create_gaussian_kernel(kernel_size)
    filtered_image = convolution(image, kernel)

    return filtered_image

def custom_min(arr):
    # Manually find the minimum value in the array
    min_val = arr[0, 0]
    for row in arr:
        for val in row:
            if val < min_val:
                min_val = val
    return min_val

def custom_max(arr):
    # Manually find the maximum value in the array
    max_val = arr[0, 0]
    for row in arr:
        for val in row:
            if val > max_val:
                max_val = val
    return max_val

def erosion(img, kernel_size):
    #kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img2=cv2.erode(img, kernel, iterations=1)
    #st.image(img2, caption=f"Eroded Image (Kernel Size: {kernel_size})", use_column_width=True)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a square kernel with ones
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    h, w = img.shape
    kh, kw = kernel.shape

    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            neighborhood = img[max(0, i - kh // 2):min(h, i + kh // 2 + 1),
                               max(0, j - kw // 2):min(w, j + kw // 2 + 1)]

            result[i, j] = np.min(neighborhood)

    return result

def dilatation(img, kernel):
    kernel = np.ones((kernel, kernel), np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2)
    #img2=cv2.dilate(img, kernel, iterations=1)
    #st.image(img2, caption=f"Eroded Image (Kernel Size: {kernel})", use_column_width=True)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    kh, kw = kernel.shape

    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            neighborhood = img[max(0, i - kh // 2):min(h, i + kh // 2 + 1), 
                               max(0, j - kw // 2):min(w, j + kw // 2 + 1)]
            result[i, j] = custom_max(neighborhood)
   

    return result

def ouverture_filtre(img, kernel_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_eroded = erosion(img, kernel_size)
    img_ouvert = dilatation(img_eroded, kernel_size)
    return img_ouvert


def fermeture_filtre(img, kernel_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_d = dilatation(img, kernel_size)
    img_ferm = erosion(img_d, kernel_size)
    return img_ferm


def filtreMediane(img, vois):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #filtered_image=cv2.medianBlur(img, (3, 3))
    #st.image(filtered_image, caption=f"Filtered Image (Average Filter)", use_column_width=True)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    imgMed = np.zeros(img.shape, img.dtype)

    for y in range(h):
        for x in range(w):
            if y < vois / 2 or y > (h - vois) / 2 or x < vois / 2 or x > (w - vois) / 2:
                imgMed[y, x] = img[y, x]
            else:
                imgVois = img[int(y - vois / 2):int(y + vois / 2), int(x - vois / 2):int(x + vois / 2)]
                imgMed[y, x] = findMedian(imgVois)

    return imgMed

def findMedian(arr):
    arr_flat = arr.flatten()
    length = len(arr_flat)
    median_index = length // 2
    
    return nthSmallest(arr_flat, median_index)

def nthSmallest(arr, k):
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    equal = [x for x in arr if x == pivot]

    if k < len(left):
        return nthSmallest(left, k)
    elif k < len(left) + len(equal):
        return equal[0]
    else:
        return nthSmallest(right, k - len(left) - len(equal))

#--------------------------------laplacien filter----------------------------------
def laplacian_filter(img):
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # filtered_image=cv2.Laplacian(img, cv2.CV_64F)
   # filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
   # st.image(filtered_image, caption="Filtered Image (Laplacian Filter)", use_column_width=True)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            laplacian_value = 4 * img[i, j] - img[i - 1, j] - img[i + 1, j] - img[i, j - 1] - img[i, j + 1]
            result[i, j] = laplacian_value

    return result

