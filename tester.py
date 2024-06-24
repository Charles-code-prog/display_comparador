import cv2
import numpy as np

def detect_and_compute(image):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Inicializa o detector ORB
    orb = cv2.ORB_create()
    
    # Detecta os keypoints e calcula os descritores
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    # Cria o objeto BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Faz a correspondência dos descritores
    matches = bf.match(descriptors1, descriptors2)
    
    # Ordena as correspondências pela distância
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_distances(matches, keypoints1, keypoints2):
    distances_img1 = []
    distances_img2 = []
    
    for match in matches:
        # Obtem os keypoints correspondentes
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt
        
        # Calcula a distância euclidiana entre os keypoints
        distance = np.linalg.norm(np.array(kp1) - np.array(kp2))
        
        # Adiciona as distâncias às listas
        distances_img1.append(distance)
        distances_img2.append(distance)
    
    return distances_img1, distances_img2

def main():
    # Carregar as imagens
    img1 = cv2.imread('imagem_capturada1.jpg')
    img2 = cv2.imread('rotate.jpg')
    
    # Detectar e computar os keypoints e descritores
    keypoints1, descriptors1 = detect_and_compute(img1)
    keypoints2, descriptors2 = detect_and_compute(img2)
    
    # Realizar a correspondência dos keypoints
    matches = match_keypoints(descriptors1, descriptors2)
    
    # Calcular as distâncias entre os keypoints correspondentes
    distances_img1, distances_img2 = calculate_distances(matches, keypoints1, keypoints2)
    
    # Imprimir as distâncias
    print("Distâncias entre keypoints na imagem 1:", distances_img1)
    print("Distâncias entre keypoints na imagem 2:", distances_img2)
    
    # Exibir as imagens com correspondências
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Correspondências', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
