import cv2
import time
import numpy as np
from sklearn.linear_model import LinearRegression

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3, 7, 11, 15, 20, 30])
model = LinearRegression().fit(X, y)

history = []

def get_people_count():
    try:
        with open("count.txt", "r") as f:
            return int(f.read())
    except:
        return 0

def get_queue_data():
    count = get_people_count()
    wait_time = int(model.predict([[count]])[0])
    timestamp = int(time.time())
    history.append({"time": timestamp, "count": count})
    if len(history) > 30:
        history.pop(0)

    alert = "⚠️ Overcrowded! Please attend quickly!" if count > 5 else "✅ Queue under control."

    return {
        "people_count": count,
        "predicted_wait_time": wait_time,
        "alert": alert,
        "history": history
    }

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        with open("count.txt", "w") as f:
            f.write(str(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
