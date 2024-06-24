from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def resize_image(img):
    # Obter largura e altura da imagem original
    largura, altura = img.size
    # Redimensionar a imagem para metade do tamanho
    img_resized = img.resize((largura // 2, altura // 2))
    return img_resized

def crop_img(img):
    w,h = img.size
    img = img.crop((w//2+w//4,0,w,h//2-h//4))
    return img 

def img_binary(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img_bin

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

# Carregar a imagem
imagem1 = Image.open('1c.jpg')
imagem2 = Image.open('1e.jpg')

# Redimensionar a imagem
imagem1 = resize_image(imagem1)
imagem2 = resize_image(imagem2)

# Cortar Imagem
#imagem1 = crop_img(imagem1)
#imagem2 = crop_img(imagem2)

# Binarizar
imagem1 = img_binary(imagem1)
imagem2 = img_binary(imagem2)

# Criar a figura e os subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

valeu_mse = round(mse(imagem1,imagem2),2)
value_psnr = round(cv2.PSNR(imagem1,imagem2),2)
value_ssim = round(ssim(imagem1,imagem2),2)
# Adicionar a legenda única
fig.suptitle(f'Similaridade: MSE: {valeu_mse}% | PSNR: {value_psnr}% | SSIM: {value_ssim}%', fontsize=16)

imgs = [imagem1,imagem2]
j = 0
for i in imgs:
    axs[j].imshow(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
    axs[j].axis('off')
    axs[j].set_title(f'Imagem {j} Redimensionada')   
    j = j+1 

# Ajustar o layout para evitar sobreposição
plt.tight_layout()
# Ajustar a posição da legenda após tight_layout
plt.subplots_adjust(top=0.85)

# Mostrar a figura
plt.show()
