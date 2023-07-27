from flask import Flask, render_template, Response
import cv2
import numpy as np
import PoseModule as pm

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        img = detector.findPose(img, False)
        ImList = detector.findPosition(img, False)

        if len(ImList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            color = (255, 0, 0)
            bar = np.interp(angle, (210, 310), (100, 650))
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(bar)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Draw Curl Count
            count = 0  # You should have a variable that keeps track of the curl count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 676), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
