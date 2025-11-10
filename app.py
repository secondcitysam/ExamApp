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
import csv
from werkzeug.utils import secure_filename


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

@app.route('/adminSignup')
def adminSignup():
    return render_template('admin_signup.html')

@app.route('/adminSignupAction', methods=['POST'])
def adminSignupAction():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO students (name, email, password, role) VALUES (%s, %s, %s, 'ADMIN')", (name, email, password))
    mysql.connection.commit()
    cur.close()
    flash("Admin account created successfully. Please login.", "success")
    return redirect(url_for('main'))

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

    id, name, email, pwd, role, admin_id = data
    studentInfo = {
        "Id": id,
        "Name": name,
        "Email": email,
        "Password": pwd,
        "Role": role,
        "admin_id": admin_id
    }
    session['user'] = studentInfo

    if role == 'STUDENT':
        utils.Student_Name = name
        return redirect(url_for('selectExamPage'))
    else:
        return redirect(url_for('adminStudents'))




@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('login.html')


@app.route('/uploadExam')
def uploadExam():
    guard = require_login()
    if guard:
        return guard

    # Check with the correct key name (capital 'R')
    if studentInfo.get("Role") != "ADMIN":
        flash("Unauthorized access", "error")
        return redirect(url_for('main'))

    return render_template('upload_exam.html')



@app.route('/uploadExamAction', methods=['POST'])
def uploadExamAction():
    guard = require_login()
    if guard:
        return guard

    exam_name = request.form['exam_name']
    csv_file = request.files['csv_file']
    admin_id = studentInfo['Id']

    if csv_file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('uploadExam'))

    filename = secure_filename(csv_file.filename)
    save_dir = os.path.join(app.root_path, "static", "ExamCSVs")  # ✅ absolute path
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    csv_file.save(filepath)

    print(f"📁 Saved CSV to: {filepath}")
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO exams (admin_id, exam_name, csv_filename) VALUES (%s, %s, %s)", (admin_id, exam_name, filename))
    exam_id = cur.lastrowid
    print(f"🧾 New exam inserted with ID={exam_id}")

    inserted, skipped = 0, 0

    try:
        with open(filepath, "r", encoding="utf-8-sig", newline='') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            print("🧩 CSV Headers:", headers)

            for row in reader:
                print("ROW:", row)
                if len(row) < 6:
                    print("⚠️ Skipping malformed row:", row)
                    skipped += 1
                    continue

                title, a, b, c, d, ans = [r.strip() for r in row]
                cur.execute("""
                    INSERT INTO questions (exam_id, title, choice_a, choice_b, choice_c, choice_d, correct_answer)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (exam_id, title, a, b, c, d, ans))
                inserted += 1

        mysql.connection.commit()
        cur.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"⚠️ Error inserting questions: {e}", "error")
        return redirect(url_for('uploadExam'))

    print(f"✅ Inserted {inserted} questions, Skipped {skipped}")

    flash(f"Exam uploaded successfully! Inserted {inserted} questions.", "success")
    return redirect(url_for('adminResults'))


@app.route('/selectExamPage')
def selectExamPage():
    guard = require_login()
    if guard: return guard

    if studentInfo["Role"] != "STUDENT":
        flash("Only students can access this page", "error")
        return redirect(url_for('main'))

    admin_id = studentInfo.get("admin_id")
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM exams WHERE admin_id=%s", (admin_id,))
    exams = cur.fetchall()
    cur.close()

    return render_template('exam_selection.html', exams=exams)

@app.route('/selectExam/<int:exam_id>', methods=['POST'])
def selectExam(exam_id):
    session['selected_exam'] = exam_id
    print(f"✅ Exam {exam_id} stored in session.")
    return redirect(url_for('rules'))  # ✅ Go to rules first



@app.route('/getExamQuestions')
def getExamQuestions():
    guard = require_login()
    if guard:
        return guard

    # ✅ Always check both query and session
    exam_id = request.args.get("exam_id", type=int)
    if not exam_id:
        exam_id = session.get("selected_exam")

    print(f"🧩 DEBUG: Requested questions for exam_id = {exam_id}")

    if not exam_id:
        print("❌ No exam_id found in session or query.")
        return jsonify([])

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT title, choice_a, choice_b, choice_c, choice_d, correct_answer
        FROM questions WHERE exam_id=%s
    """, (exam_id,))
    rows = cur.fetchall()
    cur.close()

    print(f"📊 DEBUG: Found {len(rows)} questions in DB for exam_id={exam_id}")

    questions = [{
        "title": r[0],
        "choices": [r[1], r[2], r[3], r[4]],
        "answer": r[5]
    } for r in rows]

    return jsonify(questions)


@app.route('/submitExam', methods=['POST'])
def submitExam():
    guard = require_login()
    if guard:
        return guard

    data = request.get_json()
    exam_id = data.get("exam_id")
    answers = data.get("answers", {})

    cur = mysql.connection.cursor()
    cur.execute("SELECT title, correct_answer FROM questions WHERE exam_id=%s", (exam_id,))
    rows = cur.fetchall()
    cur.close()

    total_questions = len(rows)
    correct = 0

    # Compare answers
    for idx, (title, correct_answer) in enumerate(rows):
        chosen = answers.get(str(idx)) or answers.get(idx)
        if chosen and chosen.strip().lower() == correct_answer.strip().lower():
            correct += 1

    total_marks = correct * 1  # 1 mark each
    percentage = round((correct / total_questions) * 100, 2) if total_questions else 0
    status = "PASS" if percentage >= 50 else "FAIL"
    date = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save to result.json
    import json, utils
    rid = utils.get_resultId()
    result_entry = {
        "Id": rid,
        "Name": studentInfo["Name"],
        "ExamName": utils.Student_Name,
        "ExamId": exam_id,
        "Marks": total_marks,
        "Total": total_questions,
        "Percentage": percentage,
        "Status": status,
        "Date": date,
        "Profile": utils.Student_Name + "_Profile.jpg"
    }

    try:
        if not os.path.exists("result.json"):
            with open("result.json", "w") as f:
                json.dump([], f)
        with open("result.json", "r+") as f:
            data = json.load(f)
            data.append(result_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    except Exception as e:
        print("⚠️ Failed to write result.json:", e)

    return jsonify({
        "message": "Exam submitted successfully",
        "marks": total_marks,
        "total": total_questions,
        "percentage": percentage,
        "status": status,
        "rid": rid
    })


@app.route('/rules')
def rules():
    guard = require_login()
    if guard: return guard

    exam_id = session.get('selected_exam')
    return render_template('ExamRules.html', exam_id=exam_id)

@app.route('/faceInput')
def faceInput():
    guard = require_login()
    if guard: return guard

    exam_id = session.get('selected_exam')
    return render_template('ExamFaceInput.html', exam_id=exam_id)

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

    exam_id = session.get('selected_exam')
    try:
        utils.fr.encode_faces()
    except Exception as e:
        print(f"Encode faces error: {e}")
    return render_template('ExamConfirmFaceInput.html', profile=profileName, exam_id=exam_id)

@app.route('/systemCheck')
def systemCheck():
    guard = require_login()
    if guard: return guard

    exam_id = session.get('selected_exam')
    return render_template('ExamSystemCheck.html', exam_id=exam_id)

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

    # ✅ Get exam ID from session (set by /selectExam)
    exam_id = session.get('selected_exam')
    if not exam_id:
        flash("No exam selected. Please choose an exam first.", "error")
        return redirect(url_for('selectExamPage'))

    # ✅ Fetch exam name for display
    cur = mysql.connection.cursor()
    cur.execute("SELECT exam_name FROM exams WHERE id=%s", (exam_id,))
    exam = cur.fetchone()
    cur.close()

    if not exam:
        flash("Exam not found.", "error")
        return redirect(url_for('selectExamPage'))

    exam_name = exam[0]

    # -----------------------
    # ✅ Proctoring logic (unchanged)
    # -----------------------
    init_camera()
    keyboard.hook(utils.shortcut_handler)
    utils.Globalflag = True
    utils.Student_Name = studentInfo["Name"]

    print(f"🟢 Starting proctoring for {utils.Student_Name} (Exam ID: {exam_id})")

    def start_proctoring_threads():
        try:
            print("🧠 Launching detection threads...")
            utils.executor.submit(utils.fr.run_recognition)
            utils.executor.submit(utils.cheat_Detection1)
            utils.executor.submit(utils.cheat_Detection2)
            utils.executor.submit(utils.a.record)
            utils.executor.submit(utils.objectDetection)
            print("✅ Detection threads launched successfully.")
        except Exception as e:
            print(f"⚠️ Failed to start proctoring threads: {e}")

    threading.Thread(target=start_proctoring_threads, daemon=True).start()

    # ✅ Return template with exam_id + name
    return render_template('Exam.html', exam_id=exam_id, exam_name=exam_name, student_name=studentInfo["Name"])



# ---------------------------
# Exam Submit
# ---------------------------
import csv
from flask import jsonify

@app.route('/exam/<int:exam_id>')
def examPage(exam_id):
    guard = require_login()
    if guard:
        return guard

    session['selected_exam'] = exam_id  # store in session for fallback

    # Optional validation
    cur = mysql.connection.cursor()
    cur.execute("SELECT exam_name FROM exams WHERE id=%s", (exam_id,))
    exam = cur.fetchone()
    cur.close()

    if not exam:
        flash("Exam not found.", "error")
        return redirect(url_for('selectExamPage'))

    # ✅ Pass exam_id and exam_name to template
    return render_template("Exam.html", exam_id=exam_id, exam_name=exam[0])



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
    if guard:
        return guard

    name = request.form.get('username', '')
    email = request.form.get('email', '')
    password = request.form.get('password', '')

    # Get currently logged-in admin's ID
    admin_id = studentInfo['Id']

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO students (Name, Email, Password, Role, admin_id)
        VALUES (%s, %s, %s, 'STUDENT', %s)
    """, (name, email, password, admin_id))
    mysql.connection.commit()
    cur.close()

    flash("Student added successfully and linked to your account!", "success")
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

