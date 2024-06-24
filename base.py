import time
import cv2 # type: ignore
import numpy as np # type: ignore


img = cv2.imread("samples/imagem_capturada1.jpg")

#cv2.imshow("teste",img)

h, w = img.shape[1], img.shape[0]
img = img[h // 8 : h // 2 + 110, w // 4 + 20 : w // 2 + 320]

print(h,w)

## Brilho e Contraste
nova_img = np.zeros(img.shape, img.dtype)
contrast = 6.0
brilho   = 6.0
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            nova_img[y,x,c] = np.clip(contrast*img[y,x,c] + brilho, 0 , 255)
    #        nova_img[y,x] = np.clip(contrast*img[y,x] + brilho, 0 , 255)


nova_img = cv2.cvtColor(nova_img, cv2.COLOR_BGR2GRAY)

_, img = cv2.threshold(nova_img, 127, 200, cv2.THRESH_BINARY)

#cv2.imshow("base",nova_img)
cv2.imshow("binary",img)
cv2.imwrite("samples/contraste.jpg",img)
print(len(img.shape))
cv2.waitKey(0)
cv2.destroyAllWindows()