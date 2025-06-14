from flask import Flask, render_template, Response, jsonify
import cv2
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import threading

app = Flask(__name__)

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3, 7, 11, 15, 20, 30])  # Wait time in minutes
model = LinearRegression().fit(X, y)

history = []

camera = cv2.VideoCapture(0)
camera_on = True
camera_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

def detect_faces():
    global camera, camera_on
    while True:
        with camera_lock:
            if not camera_on or camera is None or not camera.isOpened():
                # If camera is off, yield a blank frame
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
                continue

            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            people_count = len(faces)

            with open("count.txt", "w") as f:
                f.write(str(people_count))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    try:
        with open("count.txt", "r") as f:
            count = int(f.read().strip())
    except:
        count = 0

    wait_time = int(model.predict([[count]])[0])
    timestamp = int(time.time())

    history.append({"time": timestamp, "count": count})
    if len(history) > 50:
        history.pop(0)

    alert = "⚠️ Overcrowded! Please attend quickly!" if count > 5 else "✅ Queue under control."

    return jsonify({
        "people_count": count,
        "predicted_wait_time": wait_time,
        "alert": alert,
        "history": history
    })

@app.route('/stop')
def stop_camera():
    global camera, camera_on
    with camera_lock:
        camera_on = False
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
    return "Camera released and feed stopped."

@app.route('/start')
def start_camera():
    global camera, camera_on
    with camera_lock:
        if camera is None or not (hasattr(camera, "isOpened") and camera.isOpened()):
            camera = cv2.VideoCapture(0)
        camera_on = True
    return "Camera started and feed resumed."

if __name__ == '__main__':
    app.run(debug=True)