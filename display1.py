from PreProcess import PreProcess
from PIL import Image
import time

inicio = time.time()

def resize_image(img):
    largura, altura = img.size
    img_resized = img.resize((largura // 2, altura // 2))
    return img_resized

def crop_img(img):
    w, h = img.size
    img = img.crop((w // 2 + w // 4, 0, w, h // 2 - h // 4))
    return img 

# Carregar as imagens
imagem1, imagem2 = Image.open('1c.jpg'), Image.open('1e.jpg')

imagem1, imagem2 = resize_image(imagem1), resize_image(imagem2) 

imagem1, imagem2 = crop_img(imagem1), crop_img(imagem2) 

# Criar uma instância da classe PreProcess
preprocessor = PreProcess(imagem1, imagem2)

# Processar as imagens (redimensionar e binarizar)
preprocessor.process_images()

# Exibir as imagens e as métricas de similaridade
preprocessor.display_images(0)

fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
