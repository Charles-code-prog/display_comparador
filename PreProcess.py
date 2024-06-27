
import matplotlib.pyplot as plt # type: ignore
from skimage.metrics import structural_similarity as ssim # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import threading
import time

inicio = time.time()

class PreProcess:
    def __init__(self, img1, img2, contrast, brilho):
        self.img1 = img1
        self.img2 = img2
        self.contrast = contrast
        self.brilho   = brilho
        
    def brightness(self, img, contrast, brilho):
        nova_img = np.zeros(img.shape, img.dtype)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    nova_img[y,x,c] = np.clip(contrast * img[y,x,c] + brilho, 0 , 255)
        return nova_img
    
    def img_binary(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img_bin

    def mse(self, imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err
    

    def process_images(self):
        # self.img1 = self.resize_image(self.img1)
        # self.img2 = self.resize_image(self.img2)
        self.img1 = self.brightness(self.img1, self.contrast, self.brilho)
        #self.img2 = self.brightness(self.img2, self.contrast, self.brilho)
        self.img1 = self.img_binary(self.img1)
        cv2.imwrite("samples/processado_r.jpg",self.img1)
        self.img2 = self.img_binary(self.img2)
        
    ## Funções de Algoritmo ORB    
    def detect_and_compute(self, image):
        
        # Converte a imagem para escala de cinza
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Inicializa o detector ORB
        orb = cv2.ORB_create()
        
        # Detecta os keypoints e calcula os descritores
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_keypoints(self, descriptors1, descriptors2):
        # Cria o objeto BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Faz a correspondência dos descritores
        matches = bf.match(descriptors1, descriptors2)
        
        # Ordena as correspondências pela distância
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def calculate_matching_percentage(self, matches, keypoints1, keypoints2):
        # Calcula a porcentagem de correspondências válidas
        num_keypoints1 = len(keypoints1)
        num_keypoints2 = len(keypoints2)
        num_matches = len(matches)
        
        # A porcentagem de correspondência é a razão entre o número de correspondências e o menor número de keypoints
        matching_percentage = (num_matches / min(num_keypoints1, num_keypoints2)) * 100
        return matching_percentage

    def ORB_run(self):
        # Detectar e computar os keypoints e descritores
        keypoints1, descriptors1 = self.detect_and_compute(self.img1)
        keypoints2, descriptors2 = self.detect_and_compute(self.img2)
        
        # Realizar a correspondência dos keypoints
        matches = self.match_keypoints(descriptors1, descriptors2)
        
        # Calcular a porcentagem de correspondência
        matching_percentage = self.calculate_matching_percentage(matches, keypoints1, keypoints2)
        
        # Imprimir a porcentagem de correspondência
        print(f"Porcentagem de correspondência: {matching_percentage:.2f}%")
        
        # Exibir as imagens com correspondências
        #matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matching_percentage
        
    def display_images(self, exibir):
        value_mse = round(self.mse(self.img1, self.img2), 2)
        value_psnr = round(cv2.PSNR(self.img1, self.img2), 2)
        value_ssim = round(ssim(self.img1, self.img2), 2)
        
        status = "Aprovado" if value_mse <= 10.0 else "Reprovado"
        fator_inclinado = self.ORB_run()
        print(f'Status:{status}\nSimilaridade: MSE: {value_mse}% | ORB: {fator_inclinado:.2f} | PSNR: {value_psnr}% | SSIM: {value_ssim}%')
        
        self.exibir = exibir
        if (exibir == 1):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            imgs = [self.img1, self.img2]
            titles = ["Imagem 1 Capturada", "Imagem 2 Base"]
            
            fig.suptitle(f'Status:{status}\nSimilaridade: MSE: {value_mse}% | PSNR: {value_psnr}% | SSIM: {value_ssim}%', fontsize=16)
            
            for j in range(2):
                axs[j].imshow(cv2.cvtColor(imgs[j], cv2.COLOR_BGR2RGB))
                axs[j].axis('off')
                axs[j].set_title(titles[j])

            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.show()
        return status

