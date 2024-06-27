from PreProcess import PreProcess
from PIL import Image # type: ignore
import time
import cv2 # type: ignore
import numpy as np # type: ignore

inicio = time.time()

def resize_image(img):
    largura, altura = img.size
    img_resized = img.resize((largura * 2, altura * 2))
    return img_resized

def crop_img(img):
    h, w = img.shape[1], img.shape[0]
    img = img[h // 8 : h // 2 + 110, w // 4 + 20 : w // 2 + 320]

    return img 

# Carregar as imagens
imagem1, imagem2 = cv2.imread('samples/imagem_capturada1.jpg'), cv2.imread('samples/processado.jpg')


def rotate_image(image, angle):
    # Pega a altura e largura da imagem
    (h, w) = image.shape[:2]
    # Define o ponto central da imagem
    center = (w / 2, h / 2)
    # Obtém a matriz de rotação
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Aplica a rotação à imagem
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
    
imagem1 = rotate_image(imagem1, -15)

#imagem1, imagem2 = resize_image(imagem1), resize_image(imagem2) 

imagem1 = crop_img(imagem1) 
#imagem2 = crop_img(imagem2) 

# Criar uma instância da classe PreProcess
preprocessor = PreProcess(imagem1, imagem2, 6.0, 6.0)

# Processar as imagens (redimensionar e binarizar)
preprocessor.process_images()

# Exibir as imagens e as métricas de similaridade
preprocessor.display_images(0) # 0: não mostra imagem ; 1: mostra a comparação

fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
