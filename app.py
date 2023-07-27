from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def pose_estimation():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while True:
        success, img = cap.read()

        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(pose_estimation(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
