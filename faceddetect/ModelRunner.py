import time
from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(prototxt, model)
maskNet = load_model("mask_detector.keras")

lock = threading.Lock()


def detect_and_predict_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    timer=0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))
            print(f"startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    elif len(locs) == 0:
        print("No face detected")
    return (locs, preds)
def generate():
    while True:
        frame_processed = get_frame()
        if frame_processed is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n\r\n')



def process_frames():
    global frame
    global video_source
    while True:
        ret, frame = video_source.read()

        if not ret or frame is None:
            # If the frame is empty or not successfully retrieved, wait for a moment
            print("Please let me sleep as no work is there.......zzz...zzz....zzz.")
            time.sleep(10)

        frame = cv2.resize(frame, (600, int(frame.shape[0] * 600 / frame.shape[1])))

        (locs, preds) = detect_and_predict_mask(frame)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            print(label, end="\n")
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        with lock:
            frame = frame
        generate()



def get_frame():
    global frame
    with lock:
        return frame


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
if __name__ == '__main__':
    video_source = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    thread = threading.Thread(target=process_frames)
    thread.daemon = True
    thread.start()
    app.run(debug=False)


def process_frames():
    global frame
    global video_source
    while True:
        ret, frame = video_source.read()
        print("Frame processed")
        # rest of your code
