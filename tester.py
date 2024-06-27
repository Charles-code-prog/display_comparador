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

def calculate_matching_percentage(matches, keypoints1, keypoints2):
    # Calcula a porcentagem de correspondências válidas
    num_keypoints1 = len(keypoints1)
    num_keypoints2 = len(keypoints2)
    num_matches = len(matches)
    
    # A porcentagem de correspondência é a razão entre o número de correspondências e o menor número de keypoints
    matching_percentage = (num_matches / min(num_keypoints1, num_keypoints2)) * 100
    return matching_percentage

def main():
    # Carregar as imagens
    img1 = cv2.imread('samples/processado_r.jpg')
    img2 = cv2.imread('samples/processado.jpg')
    
    # Detectar e computar os keypoints e descritores
    keypoints1, descriptors1 = detect_and_compute(img1)
    keypoints2, descriptors2 = detect_and_compute(img2)
    
    # Realizar a correspondência dos keypoints
    matches = match_keypoints(descriptors1, descriptors2)
    
    # Calcular a porcentagem de correspondência
    matching_percentage = calculate_matching_percentage(matches, keypoints1, keypoints2)
    
    # Imprimir a porcentagem de correspondência
    print(f"Porcentagem de correspondência: {matching_percentage:.2f}%")
    
    # Exibir as imagens com correspondências
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Correspondências', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
