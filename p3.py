import os
import csv
import json
import torch
import datetime
import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDING_DIR = os.path.join(BASE_DIR, "FaceEmbeddings")
EMBEDDING_PATH = os.path.join(EMBEDDING_DIR, "embeddings.json")
STUDENT_CSV = os.path.join(BASE_DIR, "StudentDetails", "studentdetails.csv")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STUDENT_CSV), exist_ok=True)

# ---------------- FASTAPI ----------------
app = FastAPI(title="DL Face Recognition Attendance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ---------------- UTILS ----------------
def load_embeddings():
    if os.path.exists(EMBEDDING_PATH):
        with open(EMBEDDING_PATH, "r") as f:
            return json.load(f)
    return {}

def save_embeddings(data):
    with open(EMBEDDING_PATH, "w") as f:
        json.dump(data, f)

def get_embedding(face):
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        return facenet(face).cpu().numpy()[0]

# ---------------- REGISTER ----------------
@app.post("/register/")
def register(
    enrollment: str = Form(...),
    name: str = Form(...),
    num_images: int = Form(20)
):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise HTTPException(500, "Camera not available")

    embeddings = load_embeddings()
    embeddings.setdefault(enrollment, [])

    saved = 0

    try:
        while saved < num_images:
            ret, frame = cam.read()
            if not ret:
                continue

            # Detect face
            face = mtcnn(frame)

            if face is not None:
                # Generate embedding ONLY if face is detected
                emb = get_embedding(face)
                embeddings[enrollment].append(emb.tolist())
                saved += 1

                # Visual feedback
                cv2.putText(
                    frame,
                    f"Captured: {saved}/{num_images}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Show live video
            cv2.imshow("Register Face - Look at Camera", frame)

            # Allow manual quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    # Save student name mapping
    students = {}
    if os.path.exists(STUDENT_CSV):
        with open(STUDENT_CSV) as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    students[row[0]] = row[1]

    students[enrollment] = name
    with open(STUDENT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(students.items())

    save_embeddings(embeddings)

    return {
        "status": "success",
        "embeddings_saved": saved
    }

# ---------------- ATTENDANCE ----------------


@app.post("/mark-attendance/")
def mark_attendance(threshold: float = 0.7):
    embeddings = load_embeddings()
    if not embeddings:
        raise HTTPException(400, "No trained embeddings")

    # Load student names
    students = {}
    if os.path.exists(STUDENT_CSV):
        with open(STUDENT_CSV) as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    students[row[0]] = row[1]

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise HTTPException(500, "Camera not available")

    marked = {}

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            face = mtcnn(frame)

            if face is not None:
                emb = get_embedding(face)

                for sid, stored in embeddings.items():
                    sims = cosine_similarity([emb], stored)[0]

                    if max(sims) > threshold and sid not in marked:
                        marked[sid] = students.get(sid)
                        now = datetime.datetime.now()

                        file = os.path.join(
                            ATTENDANCE_DIR,
                            f"attendance_{now.date()}.csv"
                        )

                        with open(file, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [sid, students.get(sid), now]
                            )

            # Show marked names on screen
            if marked:
                cv2.putText(
                    frame,
                    "Marked: " + ", ".join(marked.values()),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Attendance - Press Q to Stop", frame)

            # Stop when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    return {"marked": marked}

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
