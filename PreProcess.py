
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import threading
import time

inicio = time.time()

class PreProcess:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

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
        self.img1 = self.img_binary(self.img1)
        self.img2 = self.img_binary(self.img2)

    def display_images(self, exibir):
        value_mse = round(self.mse(self.img1, self.img2), 2)
        value_psnr = round(cv2.PSNR(self.img1, self.img2), 2)
        value_ssim = round(ssim(self.img1, self.img2), 2)
        
        status = "Aprovado" if value_mse <= 10.0 else "Reprovado"
        
        print(f'Status:{status}\nSimilaridade: MSE: {value_mse}% | PSNR: {value_psnr}% | SSIM: {value_ssim}%')
        
        self.exibir = exibir
        if (exibir == 1):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            imgs = [self.img1, self.img2]
            titles = ["Imagem 1 Redimensionada", "Imagem 2 Redimensionada"]
            
            fig.suptitle(f'Status:{status}\nSimilaridade: MSE: {value_mse}% | PSNR: {value_psnr}% | SSIM: {value_ssim}%', fontsize=16)
            
            for j in range(2):
                axs[j].imshow(cv2.cvtColor(imgs[j], cv2.COLOR_BGR2RGB))
                axs[j].axis('off')
                axs[j].set_title(titles[j])

            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.show()
        return status

