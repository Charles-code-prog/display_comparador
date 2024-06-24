import cv2

# Inicializa a captura de vídeo da webcam (normalmente a webcam principal é o dispositivo 0)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
else:
    # Captura um quadro (frame)
    ret, frame = cap.read()

    if ret:
        # Mostra o quadro capturado
        cv2.imshow('Imagem Capturada', frame)

        # Espera até que uma tecla seja pressionada
        cv2.waitKey(0)

        # Salva o quadro capturado em um arquivo
        cv2.imwrite('samples/imagem_capturada1.jpg', frame)
        print("Imagem capturada e salva como 'imagem_capturada.jpg'.")

    else:
        print("Erro ao capturar a imagem.")

    # Libera o dispositivo de captura e fecha a janela
    cap.release()
    cv2.destroyAllWindows()
