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

def change_background(color):
    window.configure(bg=color)

# Frame oluşturma
frame = tk.Frame(window)
frame.pack(fill="both", expand=True)

# Arka planı değiştiren düğmeleri ekleyin
button1 = tk.Button(frame, text="Kirmizi", command=lambda: change_background("red"))
button1.pack(side="left", padx=10, pady=10)

button2 = tk.Button(frame, text="Mavi", command=lambda: change_background("blue"))
button2.pack(side="left", padx=10, pady=10)

button3 = tk.Button(frame, text="Yeşil", command=lambda: change_background("green"))
button3.pack(side="left", padx=10, pady=10)

image_label = tk.Label(window)
image_label.pack()

emotion_label = tk.Label(window, text="Tahmin Edilen Duygu: ")
emotion_label.pack()

# Başlat düğmesini ekleyin
start_button = tk.Button(window, text="Kamerayı Başlat", command=start_camera)
start_button.pack()

# Durdur düğmesini ekleyin
stop_button = tk.Button(window, text="Kamerayı Durdur", command=stop_camera)
stop_button.pack()

# Tkinter penceresini çalıştırma
window.mainloop()
