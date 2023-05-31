import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Modeli yükle
model = load_model('model.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def start_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()
        if not ret:
            break
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img = Image.fromarray(roi_gray)
            img_array = np.array(img)
            img_pixels = np.expand_dims(img_array, axis=0)
            img_pixels = img_pixels.astype('float32')
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            max_index = np.argmax(predictions[0])

            emotions = ('kizgin', 'iğrenmiş', 'korkmus', 'mutlu', 'üzgün', 'normal', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Yüz ifade analizi', resized_img)

        # Eş zamanlı olarak görüntüyü güncellemek için
        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)))
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def stop_camera():
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()

# Tkinter penceresini oluşturma
window = tk.Tk()
window.title("Kamera Uygulaması")
window.geometry("800x600")  # Pencere boyutunu ayarlayın
window.configure(bg="black")  # Arka planı siyah yapma

image_label = tk.Label(window)
image_label.pack()

emotion_label = tk.Label(window, text="Tahmin Edilen Duygu: ")
emotion_label.pack()

# Frame oluşturma
button_frame = tk.Frame(window, bg="black")
button_frame.pack(pady=10)

# Başlat düğmesini ekleyin
start_button = tk.Button(button_frame, text="Kamerayı Başlat", command=start_camera)
start_button.pack(side="left", padx=10)

# Durdur düğmesini ekleyin
stop_button = tk.Button(button_frame, text="Kamerayı Durdur", command=stop_camera)
stop_button.pack(side="right", padx=10)

# Tkinter penceresini çalıştırma
window.mainloop()
