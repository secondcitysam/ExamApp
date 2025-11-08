import math
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, flash
import os
from flask_mysqldb import MySQL
import json
import numpy as np
import warnings
import utils
import threading
import os
import re
import random
import time
import cv2
import keyboard
from flask import send_from_directory, abort
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, abort
import os
import re

STATIC_OUTPUT_VIDEOS = r"D:\PBL_APP\The-Online-Exam-Proctor\static\OutputVideos"

# ---------------------------
# Globals
# ---------------------------
studentInfo = None
profileName = None
camera_initialized = False
camera_active = False

# ---------------------------
# Flask & MySQL Config
# ---------------------------
warnings.filterwarnings("ignore")
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'xyz'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'kotlin2025'
app.config['MYSQL_DB'] = 'examproctordb'
mysql = MySQL(app)

executor = ThreadPoolExecutor(max_workers=4)

# inside your Flask app (main.py or app.py)

# app.py
from flask import Response, request, send_file, abort
import os, re

VIDEO_OUTPUT_DIR = r"D:\PBL_APP\The-Online-Exam-Proctor\static\OutputVideos"

@app.route('/video/<path:filename>')
def serve_video(filename):
    filepath = os.path.join(VIDEO_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return abort(404)

    file_size = os.path.getsize(filepath)
    range_header = request.headers.get('Range', None)

    if not range_header:
        # First load: give full file, Chrome can read metadata
        resp = send_file(filepath, mimetype="video/mp4", conditional=True)
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(file_size)
        return resp

    m = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if m:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
    else:
        start, end = 0, file_size - 1

    if start >= file_size:
        return abort(416)

    length = end - start + 1
    with open(filepath, "rb") as f:
        f.seek(start)
        data = f.read(length)

    resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
    resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(length)
    return resp



# Helpers
# ---------------------------
def require_login():
    global studentInfo
    if 'user' in session:
        studentInfo = session['user']
        return None
    return redirect(url_for('main'))

# ---------------------------
# Camera Management
# ---------------------------
def init_camera():
    """Initialize camera if not active."""
    global camera_initialized, camera_active
    if utils.cap is not None and utils.cap.isOpened():
        return True

    indices = [0, 1, 2, 3]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        time.sleep(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                utils.cap = cap
                camera_initialized = True
                camera_active = True
                print(f"🎥 Camera initialized at index {idx}")
                return True
        cap.release()
    print("❌ No camera found on any index.")
    return False


def release_camera():
    """Release camera safely."""
    global camera_active
    if utils.cap is not None and utils.cap.isOpened():
        utils.cap.release()
        camera_active = False
        print("📴 Camera released successfully.")

# ---------------------------
# Streaming
# ---------------------------
@app.route('/video_capture')
def video_capture():
    guard = require_login()
    if guard: return guard

    init_camera()
    def generate():
        face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        while camera_active and utils.cap and utils.cap.isOpened():
            success, frame = utils.cap.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue
            faces = face_cascade.detectMultiScale(frame, 1.2, 6)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        release_camera()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def main():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    global studentInfo
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM students WHERE Email=%s AND Password=%s", (username, password))
    data = cur.fetchone()
    cur.close()

    if data is None:
        flash('Your Email or Password is incorrect.', 'error')
        return redirect(url_for('main'))

    id, name, email, pwd, role = data
    studentInfo = {"Id": id, "Name": name, "Email": email, "Password": pwd}
    session['user'] = studentInfo

    if role == 'STUDENT':
        utils.Student_Name = name
        return redirect(url_for('rules'))
    else:
        return redirect(url_for('adminStudents'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('login.html')

@app.route('/rules')
def rules():
    guard = require_login()
    if guard: return guard
    return render_template('ExamRules.html')

@app.route('/faceInput')
def faceInput():
    guard = require_login()
    if guard: return guard
    return render_template('ExamFaceInput.html')

@app.route('/saveFaceInput')
def saveFaceInput():
    guard = require_login()
    if guard: return guard

    if not init_camera():
        flash('Camera not available.', 'error')
        return redirect(url_for('faceInput'))

    time.sleep(1)
    success, frame = utils.cap.read()
    if not success or frame is None:
        flash('Failed to capture webcam image.', 'error')
        release_camera()
        return redirect(url_for('faceInput'))

    result_id = utils.get_resultId() if hasattr(utils, "get_resultId") else random.randint(1, 999)
    global profileName
    profileName = f"{studentInfo['Name']}_{result_id:03}_Profile.jpg"
    cv2.imwrite(profileName, frame)
    utils.move_file_to_output_folder(profileName, 'Profiles')
    print(f"✅ Saved profile image {profileName}")
    release_camera()
    return redirect(url_for('confirmFaceInput'))

@app.route('/confirmFaceInput')
def confirmFaceInput():
    guard = require_login()
    if guard: return guard
    try:
        utils.fr.encode_faces()
    except Exception as e:
        print(f"Encode faces error: {e}")
    return render_template('ExamConfirmFaceInput.html', profile=profileName)

@app.route('/systemCheck')
def systemCheck():
    guard = require_login()
    if guard: return guard
    return render_template('ExamSystemCheck.html')

@app.route('/systemCheck', methods=["POST"])
def systemCheckRoute():
    guard = require_login()
    if guard: return guard
    examData = request.json or {}
    output = 'exam'
    if 'input' in examData and 'Not available' in str(examData['input']).split(';'):
        output = 'systemCheckError'
    return jsonify({"output": output})

@app.route('/systemCheckError')
def systemCheckError():
    guard = require_login()
    if guard: return guard
    return render_template('ExamSystemCheckError.html')

# ---------------------------
# Exam Start
# ---------------------------
@app.route('/exam')
def exam():
    guard = require_login()
    if guard:
        return guard

    init_camera()
    keyboard.hook(utils.shortcut_handler)
    utils.Globalflag = True
    utils.Student_Name = studentInfo["Name"]

    print(f"🟢 Starting proctoring for {utils.Student_Name}...")

    # 🚀 Start detection threads asynchronously AFTER returning the response
    def start_proctoring_threads():
        try:
            print("🧠 Launching all detection threads in background...")
            utils.executor.submit(utils.fr.run_recognition)
            utils.executor.submit(utils.cheat_Detection1)
            utils.executor.submit(utils.cheat_Detection2)
            utils.executor.submit(utils.a.record)
            utils.executor.submit(utils.objectDetection)
            print("✅ Detection threads launched successfully.")
        except Exception as e:
            print(f"⚠️ Failed to start detection threads: {e}")

    threading.Thread(target=start_proctoring_threads, daemon=True).start()

    # ✅ Immediately return the quiz page to browser
    return render_template('Exam.html')


# ---------------------------
# Exam Submit
# ---------------------------
@app.route('/exam', methods=["POST"])
def examAction():
    guard = require_login()
    if guard: return guard

    examData = request.json or {}
    exam_input = str(examData.get('input', '')).strip()
    link = ''

    if exam_input != '':
        utils.Globalflag = False
        release_camera()
        time.sleep(1)
        print("✅ Detection threads signaled to stop. Proceeding with result storage.")

        print("🔴 Proctoring stopped after submission.")

        utils.write_json({
            "Name": ('Shortcuts (' + ','.join(list(dict.fromkeys(utils.shorcuts))) + ') detected.')
                    if utils.shorcuts else "No shortcuts detected.",
            "Time": f"{len(utils.shorcuts)} Counts",
            "Mark": (1.5 * len(utils.shorcuts)),
            "RId": utils.get_resultId()
        })

        utils.shorcuts = []
        trustScore = utils.get_TrustScore(utils.get_resultId()) if os.path.exists("violation.json") else 0
        try:
            totalMark = math.floor(float(exam_input) * 6.6667)
        except:
            totalMark = 0

        if trustScore >= 30:
            status = "Fail (Cheating)"
            link = 'showResultFail'
        elif totalMark < 50:
            status = "Fail"
            link = 'showResultFail'
        else:
            status = "Pass"
            link = 'showResultPass'

        utils.write_json({
            "Id": utils.get_resultId(),
            "Name": studentInfo['Name'],
            "TotalMark": totalMark,
            "TrustScore": max(100 - trustScore, 0),
            "Status": status,
            "Date": time.strftime("%Y-%m-%d", time.localtime(time.time())),
            "StId": studentInfo['Id'],
            "Link": profileName or ''
        }, "result.json")

        resultStatus = f"{studentInfo['Name']};{totalMark};{status};{time.strftime('%Y-%m-%d', time.localtime(time.time()))}"
    else:
        utils.Globalflag = True
        resultStatus = ''

    return jsonify({"output": resultStatus, "link": link})

# ---------------------------
# Results / Admin
# ---------------------------
@app.route('/showResultPass/<result_status>')
def showResultPass(result_status):
    guard = require_login()
    if guard: return guard
    return render_template('ExamResultPass.html', result_status=result_status)

@app.route('/showResultFail/<result_status>')
def showResultFail(result_status):
    guard = require_login()
    if guard: return guard
    return render_template('ExamResultFail.html', result_status=result_status)

@app.route('/adminResults')
def adminResults():
    import utils
    try:
        results = utils.getResults()  # read from results.json
    except Exception as e:
        print(f"⚠️ Error loading results: {e}")
        results = []
    return render_template('Results.html', results=results)


@app.route('/adminResultDetails/<int:resultId>')
def adminResultDetails(resultId):
    guard = require_login()
    if guard: return guard
    try:
        result_details = utils.getResultDetails(resultId)
    except Exception as e:
        print(f"⚠️ Error loading result details for ID {resultId}: {e}")
        result_details = {"Result": [], "Violation": []}
    return render_template('ResultDetails.html', resultDetials=result_details)

@app.route('/adminResultDetailsVideo/<path:videoInfo>')
def adminResultDetailsVideo(videoInfo):
    """
    Renders the video player for recorded violation videos.
    videoInfo = "<filename>;<resultId>"
    """
    try:
        # Debug print to confirm data flow
        print(f"🎬 Opening ResultDetailsVideo.html for: {videoInfo}")
        return render_template('ResultDetailsVideo.html', videoInfo=videoInfo)
    except Exception as e:
        print(f"⚠️ Error in adminResultDetailsVideo: {e}")
        flash("Error loading video.", "error")
        return redirect(url_for('adminResults'))



@app.route('/adminStudents')
def adminStudents():
    guard = require_login()
    if guard: return guard
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM students WHERE Role='STUDENT'")
    data = cur.fetchall()
    cur.close()
    return render_template('Students.html', students=data)

@app.route('/insertStudent', methods=['POST'])
def insertStudent():
    guard = require_login()
    if guard: return guard
    name = request.form.get('username', '')
    email = request.form.get('email', '')
    password = request.form.get('password', '')
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                (name, email, password, 'STUDENT'))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('adminStudents'))

@app.route('/deleteStudent/<string:stdId>')
def deleteStudent(stdId):
    guard = require_login()
    if guard: return guard
    flash("Record deleted successfully.")
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM students WHERE ID=%s", (stdId,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('adminStudents'))

@app.route('/updateStudent', methods=['POST'])
def updateStudent():
    guard = require_login()
    if guard: return guard
    id_data = request.form.get('id', '')
    name = request.form.get('name', '')
    email = request.form.get('email', '')
    password = request.form.get('password', '')
    cur = mysql.connection.cursor()
    cur.execute("""UPDATE students SET Name=%s, Email=%s, Password=%s WHERE ID=%s""",
                (name, email, password, id_data))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('adminStudents'))

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == '__main__':
    os.makedirs('static/Profiles', exist_ok=True)
    os.makedirs('static/OutputVideos', exist_ok=True)
    os.makedirs('static/OutputAudios', exist_ok=True)
    # Add threaded=True so the generator (/video_capture) can’t starve other handlers.
    app.run(debug=True, threaded=True)

