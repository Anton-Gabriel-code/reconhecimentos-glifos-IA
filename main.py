import cv2

import numpy as np

CONFIDENCE_THRESHOLD = 0.80

CLASSES = ['Neutro', 'Água', 'Luz']

VIDEOS = { 
          'Água': 'videos/rainflinger.mp4',
          'Luz': 'videos/light_beam.mp4'
          }


print('Carregando o modelo treinado...')

model = load_model('keras_model.h5', compile=False)
data = np.empty((1, 224,224,3), dtype = np.float32)

cap = cv2.VideoCapture(0)
print('A Webcam está sendo incializada...\nMostre o glifo para lançar a mágia. (Pressione "Q" para sair.)')


def reproduzir_magia(caminho_video):
    video_player = cv2.VideoCapture(caminho_video)
    while video_player.isOpened():
        ret, frame_video = video_player.read()
        if not ret:
            break
        cv2.imshow('Canalizando...', frame_video)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    video_player.release()
    cv2.destroyWindow('Canalizando...')
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    image_resized = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_normalized = (image_resized.astype(np.float32)/ 127.5) - 1
    data[0] = image_normalized
    
    prediction = model.predict(data, verbose = 0)
    index = np.argmax(prediction)
    classe_atual = CLASSES[index]
    confianca = prediction[0][index]
    
    texto_tela = f'{classe_atual}: {confianca * 100:.2f}%'
    cv2.putText(frame, texto_tela, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Grimorio de Visão Computacional', frame)
    
    
    if confianca >= CONFIDENCE_THRESHOLD and classe_atual in VIDEOS:
        print(f'Feitiço detectado: {classe_atual} ({confianca * 100:.1f}%)!\n Canalizando Feitiço...')
        reproduzir_magia(VIDEOS[classe_atual])
        
    if cv2.waitKey(1) & 0xff == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()