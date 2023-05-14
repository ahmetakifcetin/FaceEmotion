import cv2
from deepface import DeepFace

def recognize_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(face, actions=['emotion'])
            emotion = result['emotion']['dominant']
        except:
            emotion = 'Unknown'
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        processed_frame = recognize_faces(frame)

        cv2.imshow('Yüz Tanıma ve Duygu Analizi', processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
