import sys
import face_recognition
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
import os
import json
import shutil
import keyboard
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO
import threading
import pyaudio
import struct
import wave
import datetime
import subprocess
import traceback

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=6)

# -------------------- Global Vars --------------------
Globalflag = False
Student_Name = ''
shorcuts = []
cap = None

# Directories
PROFILE_DIR = "static/Profiles"
VIDEO_OUTPUT_DIR = "static/OutputVideos"
AUDIO_OUTPUT_DIR = "static/OutputAudios"


import shutil
import subprocess
import tempfile

FFMPEG_BIN = "ffmpeg"  # assumes ffmpeg is on PATH

def transcode_to_h264(mp4_path: str) -> str:
    """
    Re-mux/re-encode whatever OpenCV produced into H.264 (avc1) + faststart,
    so browsers show correct duration & play reliably.
    Returns the final path (same as input).
    """
    # write to a temp file in same dir, then replace atomically
    dir_ = os.path.dirname(mp4_path)
    base = os.path.basename(mp4_path)
    temp_out = os.path.join(dir_, f".__tmp_{base}")

    # re-encode to AVC baseline (browser safe), keep 10 fps, make it web-safe
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", mp4_path,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-r", "10",
        "-movflags", "+faststart",
        "-an",
        temp_out
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print("⚠️ ffmpeg failed:", proc.stderr.decode(errors="ignore")[:4000])
            return mp4_path
        # replace original
        os.replace(temp_out, mp4_path)
        print(f"✅ Transcoded to H.264: {os.path.basename(mp4_path)}")
    except FileNotFoundError:
        print("⚠️ ffmpeg not found on PATH. Install it or set FFMPEG_BIN path.")
    except Exception as e:
        print("⚠️ ffmpeg error:", e)
        try:
            if os.path.exists(temp_out):
                os.remove(temp_out)
        except: pass
    return mp4_path

# -------------------- Utility Functions --------------------

def write_json(new_data, filename='violation.json'):
    """Safely append new data to JSON with fallback if file empty."""
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    try:
        with open(filename, 'r+') as file:
            file_data = json.load(file)
            if isinstance(new_data, dict) and "Link" not in new_data:
                new_data["Link"] = ""
            file_data.append(new_data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except Exception as e:
        print(f"⚠️ Error writing to {filename}: {e}")

def move_file_to_output_folder(file_name, folder_name='OutputVideos'):
    """Move temp video files into static output folders."""
    current_directory = os.getcwd()
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(current_directory, 'static', folder_name, file_name)
    try:
        shutil.move(source_path, destination_path)
        print(f"✅ Moved {file_name} to {folder_name}")
    except FileNotFoundError:
        print(f"❌ File '{file_name}' not found.")
    except shutil.Error as e:
        print(f"⚠️ Move failed: {e}")

def reduceBitRate(input_file, output_file):
    """Compress video bitrate via ffmpeg."""
    ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"  # 🔧 Adjust this path
    command = [
        ffmpeg_path, "-i", input_file,
        "-b:v", "1000k",
        "-c:v", "libx264",
        "-c:a", "aac", "-b:a", "192k",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"🎞️ Bitrate reduced for {input_file}")

def capture_screen():
    """Capture current screen frame."""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_resultId():
    """Return next available Result ID."""
    if not os.path.exists('result.json'):
        with open('result.json', 'w') as f: json.dump([], f)
    with open('result.json', 'r+') as file:
        data = json.load(file)
        if not data: return 1
        data.sort(key=lambda x: x.get("Id", 0))
        return data[-1]["Id"] + 1

def get_TrustScore(Rid):
    """Compute Trust Score by summing violation marks."""
    if not os.path.exists('violation.json'): return 0
    with open('violation.json', 'r') as file:
        data = json.load(file)
        filtered = [v for v in data if v.get("RId") == Rid]
        return sum(v.get("Mark", 0) for v in filtered)
# -------------------- Keyboard Shortcut Detection --------------------
import keyboard
import time

shorcuts = []  # already defined at top

def shortcut_handler(event):
    """Detect and record restricted keyboard shortcuts during the exam."""
    import utils  # safe self-import inside handler
    if event.event_type != keyboard.KEY_DOWN:
        return

    shortcut = ""
    # Common prohibited combinations
    try:
        if keyboard.is_pressed("ctrl") and keyboard.is_pressed("c"):
            shortcut = "Ctrl+C"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("v"):
            shortcut = "Ctrl+V"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("a"):
            shortcut = "Ctrl+A"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("x"):
            shortcut = "Ctrl+X"
        elif keyboard.is_pressed("alt") and keyboard.is_pressed("tab"):
            shortcut = "Alt+Tab"
        elif keyboard.is_pressed("alt") and keyboard.is_pressed("shift") and keyboard.is_pressed("tab"):
            shortcut = "Alt+Shift+Tab"
        elif keyboard.is_pressed("win") and keyboard.is_pressed("tab"):
            shortcut = "Win+Tab"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("esc"):
            shortcut = "Ctrl+Esc"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("t"):
            shortcut = "Ctrl+T"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("w"):
            shortcut = "Ctrl+W"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("z"):
            shortcut = "Ctrl+Z"
        elif keyboard.is_pressed("print_screen"):
            shortcut = "PrtSc"
        elif keyboard.is_pressed("f1"):
            shortcut = "F1"
        elif keyboard.is_pressed("f2"):
            shortcut = "F2"
        elif keyboard.is_pressed("f3"):
            shortcut = "F3"
        elif keyboard.is_pressed("win"):
            shortcut = "Windows Key"
        elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt") and keyboard.is_pressed("del"):
            shortcut = "Ctrl+Alt+Del"
    except:
        # Ignore transient key errors
        return

    if shortcut and shortcut not in shorcuts:
        shorcuts.append(shortcut)
        print(f"⚠️ Detected prohibited shortcut: {shortcut}")

        # ✅ Log this violation immediately to violation.json
        try:
            violation_entry = {
                "Name": f"Prohibited Shortcut ({shortcut}) detected",
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "Duration": "Instant",
                "Mark": 1.5,
                "Link": "",
                "RId": utils.get_resultId()
            }
            utils.write_json(violation_entry)
        except Exception as e:
            print(f"⚠️ Could not write shortcut violation: {e}")


# -------------------- Face Recognition --------------------

def face_confidence(face_distance, threshold=0.6):
    range_val = (1.0 - threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)
    if face_distance > threshold:
        return str(round(linear_val * 100, 2)) + '%'
    value = (linear_val + ((1.0 - linear_val) * ((linear_val - 0.5) * 2) ** 0.2)) * 100
    return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        """Encode known faces from static/Profiles."""
        if not os.path.exists(PROFILE_DIR):
            os.makedirs(PROFILE_DIR)
        for image in os.listdir(PROFILE_DIR):
            path = os.path.join(PROFILE_DIR, image)
            try:
                face_image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(face_image)
                if len(encodings) == 0:
                    print(f"⚠️ No face found in {image}, skipping.")
                    continue
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(image)
            except Exception as e:
                print(f"❌ Error encoding {image}: {e}")
        print(f"✅ Encoded faces: {self.known_face_names}")

    def run_recognition(self):
        """Continuously verify student via webcam."""
        global Globalflag, cap
        if cap is None:
            cap = cv2.VideoCapture(0)
        print(f"🎥 Face Verification started | Globalflag={Globalflag}")

        while Globalflag:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Empty frame, skipping...")
                time.sleep(0.2)
                continue

            text = "Verified Student disappeared"

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                face_locations = face_recognition.face_locations(rgb_small_frame)
                encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.55
                    )
                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match = np.argmin(distances) if len(distances) > 0 else None

                    if best_match is not None and matches[best_match]:
                        name = str(self.known_face_names[best_match]).split('_')[0]
                        confidence = face_confidence(distances[best_match])
                        if name == Student_Name and float(confidence[:-1]) >= 80:
                            text = "Verified Student appeared"

            self.process_current_frame = not self.process_current_frame
            print(f"🎯 {text}")
            time.sleep(0.2)

        if cap:
            cap.release()
            print("✅ Camera released after verification stop")

# -------------------- Audio Recorder --------------------

class Recorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                  input=True, output=False, frames_per_buffer=1024)
        self.timeout = 0
        self.quiet = []
        self.quiet_idx = -1

    def rms(self, frame):
        count = len(frame) / 2
        fmt = "%dh" % count
        shorts = struct.unpack(fmt, frame)
        sum_sq = sum(sample * sample for sample in shorts)
        return (sum_sq / count) ** 0.5 / 32768 * 1000

    def inSound(self, data):
        rms_val = self.rms(data)
        curr = time.time()
        if rms_val > 10:
            self.timeout = curr + 3
            return True
        if curr < self.timeout:
            return True
        self.timeout = 0
        return False

    def record(self):
        """Continuously record sounds when detected."""
        global Globalflag
        print("🎙️ Voice Recorder started")

        frames = []
        while Globalflag:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                if self.inSound(data):
                    frames.append(data)
                    # keep capturing until sound stops
                    if len(frames) >= 50:  # ~3 seconds
                        self.save_audio(frames)
                        frames = []
                else:
                    if frames:
                        self.save_audio(frames)
                        frames = []
            except Exception as e:
                print(f"⚠️ Audio read error: {e}")

        if frames:
            self.save_audio(frames)

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("🎙️ Voice Recorder stopped")

    def save_audio(self, frames):
        filename = f"{random.randint(1000,9999)}VoiceViolation.wav"
        filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
        try:
            if not os.path.exists(AUDIO_OUTPUT_DIR):
                os.makedirs(AUDIO_OUTPUT_DIR)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames))
            violation = {
                "Name": "Voice detected during exam",
                "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Duration": "few secs",
                "Mark": 1,
                "Link": filename,
                "RId": get_resultId()
            }
            write_json(violation)
            print(f"💾 Saved voice violation: {filename}")
        except Exception as e:
            print(f"⚠️ Error saving audio: {e}")

# -------------------- Global Instances --------------------


# -------------------- Background Detection Threads --------------------

# -------------------- Updated Proctoring Detection Threads --------------------
def save_video_clip(frame):
    """Save ~3s clip via OpenCV then transcode to H.264 for browser playback."""
    global cap
    filename = f"{random.randint(1000,9999)}Violation.mp4"
    filepath = os.path.join(VIDEO_OUTPUT_DIR, filename)
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    # WARNING: OpenCV mp4v is not browser friendly, but we will transcode after.
    out = cv2.VideoWriter(
        filepath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,
        (frame.shape[1], frame.shape[0])
    )
    start = time.time()
    while time.time() - start < 3 and cap is not None and cap.isOpened():
        ok, fr = cap.read()
        if not ok: break
        out.write(fr)
    out.release()
    print(f"🎞️ Saved raw clip (mp4v): {filename}")

    # 🔁 Make it web-friendly (avc1 + faststart)
    transcode_to_h264(filepath)
    return filename


# ======================================================
#  HIGH-VISIBILITY DEBUG / DIAGNOSTIC LOGGING
# ======================================================


def log_event(level, msg):
    """Rich timestamped logger with level tags."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")


import traceback

def cheat_Detection1():
    """Head movement detection with fine-grained logs and safe shutdown."""
    deleteTrashVideos()
    global Globalflag
    log_event("INFO", "🎯 [CD1] Head Movement Detection INITIATED")

    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        log_event("ERROR", "Camera failed to open in CD1. Thread aborting.")
        return

    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3,
                                          min_tracking_confidence=0.3)
        log_event("INFO", "✅ Mediapipe FaceMesh initialized successfully.")
    except Exception as e:
        log_event("ERROR", f"Failed to init Mediapipe FaceMesh: {e}")
        traceback.print_exc()
        cap_local.release()
        return

    prev_state, last_logged = None, 0

    while True:
        if not Globalflag:
            log_event("INFO", "[CD1] Globalflag=False → stopping detection thread.")
            break

        ok, frame = cap_local.read()
        if not ok or frame is None:
            log_event("WARN", "CD1: No frame read from camera.")
            time.sleep(0.2)
            continue

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                log_event("DEBUG", "CD1: No face landmarks found this frame.")
                continue

            for lm in res.multi_face_landmarks:
                nose = lm.landmark[1].x
                state = "center"
                if nose < 0.4: state = "left"
                elif nose > 0.6: state = "right"

                if prev_state and state != prev_state and time.time() - last_logged > 5:
                    log_event("ALERT", f"Head moved {prev_state} → {state}")
                    try:
                        name = save_video_clip(frame)
                        write_json({
                            "Name": f"Head Movement ({prev_state}->{state})",
                            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration": "3 sec",
                            "Mark": 2,
                            "Link": name,
                            "RId": get_resultId()
                        })
                        log_event("INFO", f"Head movement proof saved: {name}")
                    except Exception as e:
                        log_event("ERROR", f"Failed to save proof clip: {e}")
                        traceback.print_exc()
                    last_logged = time.time()
                prev_state = state

        except Exception as e:
            log_event("ERROR", f"CD1 loop error: {e}")
            traceback.print_exc()

    try:
        cap_local.release()
        log_event("INFO", "[CD1] Camera released successfully.")
    except Exception as e:
        log_event("ERROR", f"[CD1] Failed to release camera: {e}")

    log_event("INFO", "🛑 [CD1] Head Movement Detection Stopped.")


def cheat_Detection2():
    """Multi-face + YOLO object detection with pinpoint logs and safe shutdown."""
    deleteTrashVideos()
    global Globalflag
    log_event("INFO", "🎯 [CD2] Multi-Face/Object Detection INITIATED")

    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        log_event("ERROR", "Camera failed to open in CD2. Thread aborting.")
        return

    # Try loading YOLO
    try:
        model = YOLO("yolov8n.pt")
        log_event("INFO", f"✅ YOLO model loaded successfully with {len(model.names)} classes.")
    except Exception as e:
        log_event("ERROR", f"YOLO load failed: {e}")
        traceback.print_exc()
        model = None

    cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    if cascade.empty():
        log_event("ERROR", "Haarcascade XML not found or invalid!")

    last_face, last_obj = 0, 0

    while True:
        if not Globalflag:
            log_event("INFO", "[CD2] Globalflag=False → stopping detection thread.")
            break

        ok, frame = cap_local.read()
        if not ok or frame is None:
            log_event("WARN", "CD2: Empty camera frame.")
            time.sleep(0.1)
            continue

        # --- Multi-face detection ---
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            log_event("DEBUG", f"[CD2] Faces detected={len(faces)}")

            if len(faces) > 1 and time.time() - last_face > 5:
                log_event("ALERT", f"[CD2] Multiple faces ({len(faces)}) detected.")
                clip = save_video_clip(frame)
                write_json({
                    "Name": f"Multiple Faces ({len(faces)})",
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration": "3 sec",
                    "Mark": 3,
                    "Link": clip,
                    "RId": get_resultId()
                })
                log_event("INFO", f"[CD2] Multiple face proof saved: {clip}")
                last_face = time.time()
        except Exception as e:
            log_event("ERROR", f"[CD2] Face cascade error: {e}")
            traceback.print_exc()

        # --- YOLO object detection ---
        if model:
            try:
                results = model(frame, verbose=False)
                labels = [model.names[int(b.cls[0])].lower()
                          for r in results for b in r.boxes]
                log_event("DEBUG", f"[CD2] YOLO raw labels={labels}")

                hits = [l for l in labels if any(w in l for w in
                        ["cell", "phone", "laptop", "tablet", "monitor"])]
                if hits and time.time() - last_obj > 5:
                    log_event("ALERT", f"[CD2] Electronic object(s) detected={hits}")
                    clip = save_video_clip(frame)
                    write_json({
                        "Name": f"Electronic Object {hits}",
                        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Duration": "3 sec",
                        "Mark": 2.5,
                        "Link": clip,
                        "RId": get_resultId()
                    })
                    log_event("INFO", f"[CD2] YOLO proof saved: {clip}")
                    last_obj = time.time()
            except Exception as e:
                log_event("ERROR", f"[CD2] YOLO inference failure: {e}")
                traceback.print_exc()

        time.sleep(0.3)

    try:
        cap_local.release()
        log_event("INFO", "[CD2] Camera released successfully.")
    except Exception as e:
        log_event("ERROR", f"[CD2] Failed to release camera: {e}")

    log_event("INFO", "🛑 [CD2] Multi-Face/Object Detection Stopped.")


def objectDetection():
    """Standalone YOLO with extreme verbosity and safe shutdown."""
    global Globalflag
    log_event("INFO", "📱 [OD] Stand-alone Object Detection INITIATED")

    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        log_event("ERROR", "Camera failed to open in OD thread.")
        return

    try:
        model = YOLO("yolov8n.pt")
        log_event("INFO", "✅ YOLOv8 model loaded successfully in OD thread.")
    except Exception as e:
        log_event("ERROR", f"YOLO load fail in OD: {e}")
        traceback.print_exc()
        cap_local.release()
        return

    while True:
        if not Globalflag:
            log_event("INFO", "[OD] Globalflag=False → stopping detection thread.")
            break

        ok, frame = cap_local.read()
        if not ok or frame is None:
            log_event("WARN", "OD: Blank frame captured.")
            continue

        try:
            res = model(frame, verbose=False)
            labels = [model.names[int(b.cls[0])].lower()
                      for r in res for b in r.boxes]
            log_event("DEBUG", f"[OD] YOLO labels={labels}")

            targets = [l for l in labels if any(w in l for w in
                       ["cell", "phone", "laptop", "tablet", "monitor"])]
            if targets:
                log_event("ALERT", f"[OD] Electronic object(s) detected={targets}")
                name = save_video_clip(frame)
                write_json({
                    "Name": f"Electronic Object {targets}",
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration": "3 sec",
                    "Mark": 2.5,
                    "Link": name,
                    "RId": get_resultId()
                })
                log_event("INFO", f"[OD] Proof saved: {name}")
                time.sleep(3)
        except Exception as e:
            log_event("ERROR", f"[OD] YOLO inference crash: {e}")
            traceback.print_exc()

        time.sleep(0.5)

    try:
        cap_local.release()
        log_event("INFO", "[OD] Camera released successfully.")
    except Exception as e:
        log_event("ERROR", f"[OD] Failed to release camera: {e}")

    log_event("INFO", "🛑 [OD] Object Detection thread stopped.")



# -------------------- Results and Violations --------------------

def getResults():
    """Return list of all results stored in result.json."""
    try:
        with open('result.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                # Sort by latest first
                return sorted(data, key=lambda x: x.get("Id", 0), reverse=True)
            else:
                return []
    except FileNotFoundError:
        print("⚠️ result.json not found — returning empty list.")
        return []
    except Exception as e:
        print(f"⚠️ Error reading result.json: {e}")
        return []

def getResultDetails(rid: int):
    """Return detailed result and all linked violations for a given result ID."""
    try:
        with open("result.json", "r", encoding="utf-8") as rf:
            result_data = json.load(rf)
            # Filter the matching result
            filtered_result = [item for item in result_data if int(item.get("Id", -1)) == int(rid)]
    except FileNotFoundError:
        print("⚠️ result.json not found.")
        filtered_result = []
    except Exception as e:
        print(f"⚠️ Error reading result.json: {e}")
        filtered_result = []

    try:
        with open("violation.json", "r", encoding="utf-8") as vf:
            violation_data = json.load(vf)
            # Filter only violations matching this Result ID
            filtered_violations = [v for v in violation_data if int(v.get("RId", -1)) == int(rid)]
    except FileNotFoundError:
        print("⚠️ violation.json not found.")
        filtered_violations = []
    except Exception as e:
        print(f"⚠️ Error reading violation.json: {e}")
        filtered_violations = []

    return {
        "Result": filtered_result,
        "Violation": filtered_violations
    }
# -------------------- Electronic Object Detection --------------------




a = Recorder()
fr = FaceRecognition()
