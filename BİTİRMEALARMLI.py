from ultralytics import YOLO
import cv2
import pygame

# YOLO modelini yükleyin
model = YOLO('best.pt')

# pygame'i başlatın
pygame.mixer.init()

# Ses dosyasını yükleyin
sound = pygame.mixer.Sound('Alarm.mp3')  # Burada çalmak istediğiniz ses dosyasının yolunu belirtin

# Kamera kaynağını açın
cap = cv2.VideoCapture(cv2.CAP_ANY)
if cap.isOpened():
    print("Kamera başarıyla açıldı.")
else:
    print("Kamera açılamadı.")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Eğer bir hayvan tespit edilirse ses çal, tespit kaybolduğunda sesi durdur
        if results[0].boxes:  # Eğer herhangi bir tespit varsa
            if not pygame.mixer.get_busy():  # Eğer ses çalmıyorsa
                sound.play(-1)  # Ses dosyasını döngü halinde çal
        else:
            if pygame.mixer.get_busy():  # Eğer ses çalıyorsa
                sound.stop()  # Ses dosyasını durdur
        
        cv2.imshow('Area Protection Monitor', annotated_frame)
        
        if cv2.waitKey(1) == 13:  # 'Enter' tuşuna basıldığında çık
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
