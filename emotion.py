from tkinter import messagebox
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import pyttsx3

# Sesli geribildirim için pyttsx3 kullanımı
engine = pyttsx3.init()
engine.setProperty('rate', 120)

# Modeli yükle
model = load_model(r'C:\Users\alp54\PycharmProjects\EmotionDetection\best_model.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

is_sound_enabled = False  # Sesin etkinleştirilip etkinleştirilmediğini kontrol etmek için değişken
last_detected_emotion = None  # En son tespit edilen duygu bilgisini saklamak için değişken


def start_camera():
    global last_detected_emotion
    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()
        if not ret:
            break
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            center_x = x + w // 2
            center_y = y + h // 2
            radius = min(w, h) // 2
            cv2.circle(test_img, (center_x, center_y), radius, (255, 0, 0), thickness=2)
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
            last_detected_emotion = predicted_emotion

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

    # Program sonlandığında son tespit edilen duygu üzerinden sesli geribildirim sağlama
    if is_sound_enabled and last_detected_emotion is not None:
        if last_detected_emotion == 'mutlu':
           engine.say("You are happy. What a surprise!")

        elif last_detected_emotion == 'kizgin':
             engine.say("Why are you mad? Life is too short to be mad.")

        elif last_detected_emotion == 'normal':
            engine.say("You look normal. Do you have any emotions?")

        elif last_detected_emotion == "korkmus":
            engine.say("Oh my god! What did you see?")

        else:
            engine.say("Son tespit edilen duygu: " + last_detected_emotion)

        engine.runAndWait()



def toggle_sound():
    global is_sound_enabled
    is_sound_enabled = not is_sound_enabled
    sound_button_text = "Sesli Mesajı Aç" if not is_sound_enabled else "Sesli Mesajı Kapat"
    sound_button.configure(text=sound_button_text)


# Tkinter penceresini oluşturma
window = tk.Tk()
window.title("Duygu Tespit Etme Uygulaması")
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

# Sesli geribildirim düğmesini ekleyin
sound_button_text = "Sesli Mesajı Aç" if not is_sound_enabled else "Sesli Mesajı Kapat"
sound_button = tk.Button(window, text=sound_button_text, command=toggle_sound)
sound_button.pack()

# Tkinter penceresini çalıştırma
window.mainloop()
