import cv2
from deepface import DeepFace

# Yüz tanıma ve duygu analizi fonksiyonu
def recognize_faces(image):
    # Yüz tespiti
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yüz bölgesini kırpma
        face = image[y:y+h, x:x+w]

        # Yüz tanıma ve duygu analizi
        try:
            result = DeepFace.analyze(face, actions=['emotion'])
            emotion = result['emotion']['dominant']
        except:
            emotion = 'Unknown'

        # Yüz ve duygu bilgilerini görselleştirme
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Ana işlem döngüsü
def main():
    # Video akışı başlatma
    cap = cv2.VideoCapture(0)

    while True:
        # Video akışından bir çerçeve alınması
        ret, frame = cap.read()

        if not ret:
            break

        # Yüz tanıma ve duygu analizi
        processed_frame = recognize_faces(frame)

        # İşlenmiş çerçevenin gösterilmesi
        cv2.imshow('Yüz Tanıma ve Duygu Analizi', processed_frame)

        # Çıkış tuşu kontrolü
        if cv2.waitKey(1) == ord('q'):
            break

    # Kaynakları serbest bırakma
    cap.release()
    cv2.destroyAllWindows()

# Ana işlevi çağırma
if __name__ == '__main__':
    main()
