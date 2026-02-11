# Attendence-through-CV

🎓 DL Face Recognition Attendance System (FastAPI + FaceNet)
📌 Project Overview

This project is a Deep Learning–based Face Recognition Attendance System built using:

FastAPI for backend API development

FaceNet (InceptionResnetV1) for facial embedding generation

MTCNN for face detection

OpenCV for camera handling

Cosine Similarity for face matching

The system allows:

👤 Student Registration using face capture

📸 Real-time Face Recognition

📝 Automatic Attendance Marking

💾 Persistent Storage of embeddings and attendance records

🏗️ Project Architecture
<pre>
Project Root
│
├── p3.py                  # Main FastAPI application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── FaceEmbeddings/        # Stored face embeddings (JSON)
│   └── embeddings.json
│
├── StudentDetails/
│   └── studentdetails.csv
│
├── Attendance/            # Daily attendance CSV files
│   └── attendance_YYYY-MM-DD.csv
│
└── .gitignore
</pre>
🚀 Technologies & Libraries Used
1️⃣ FastAPI

Used to build REST APIs.

Provides high performance.

Automatically generates Swagger UI at:

http://localhost:8000/docs

2️⃣ OpenCV (cv2)

Accesses webcam.

Captures real-time video frames.

Displays video feed with overlay text.

3️⃣ MTCNN (Multi-task Cascaded Neural Network)

Detects faces in real-time.

Extracts aligned face from frame.

Ensures only face region is passed to FaceNet.

4️⃣ FaceNet (InceptionResnetV1 - pretrained on VGGFace2)

Converts face image into a 512-dimensional embedding vector.

Same person → Similar embeddings.

Different person → Different embeddings.

5️⃣ PyTorch

Backend framework powering FaceNet model.

Handles GPU/CPU device allocation.

6️⃣ Scikit-Learn (Cosine Similarity)

Compares new face embedding with stored embeddings.

Determines if faces match based on similarity score.

7️⃣ JSON & CSV

JSON → Stores face embeddings.

CSV → Stores student details and attendance records.

📡 API Endpoints
1️⃣ Register Student
Endpoint
POST /register/

Parameters (Form Data)
Parameter	Type	Description
enrollment	str	Unique student ID
name	str	Student name
num_images	int	Number of face samples (default: 20)
🔄 How Registration Works

Webcam starts automatically.

MTCNN detects the face in each frame.

FaceNet generates embedding vector.

Embedding is stored in:

FaceEmbeddings/embeddings.json


Student ID + Name stored in:

StudentDetails/studentdetails.csv


System captures multiple embeddings (default = 20).

This improves recognition accuracy.
<pre>
💾 How Data Is Stored
embeddings.json
{
  "101": [
    [0.123, 0.456, ...],
    [0.234, 0.567, ...]
  ],
  "102": [
    [0.987, 0.654, ...]
  ]
}
</pre>

Each student ID maps to a list of 512-dimensional vectors.

studentdetails.csv
101,John Doe
102,Alice Smith

2️⃣ Mark Attendance
Endpoint
POST /mark-attendance/

Optional Parameter
Parameter	Type	Default
threshold	float	0.7
🔄 How Attendance Works

System loads all stored embeddings.

Webcam starts.

Face detected using MTCNN.

Embedding generated using FaceNet.

Cosine similarity computed between:

Live embedding

Stored embeddings

If:

max_similarity > threshold


→ Face is recognized.

📝 Attendance Marking Process

Student ID identified.

Name retrieved from CSV.

Current date & time captured.

Entry appended to daily attendance file:

Attendance/attendance_2026-02-11.csv

Attendance CSV Format
EnrollmentID,Name,Timestamp
101,John Doe,2026-02-11 10:32:12

🧠 Duplicate Prevention

The system ensures:

Same student is not marked twice in one session.

Uses a dictionary marked to track already recorded students.

📊 How Face Matching Works

Cosine Similarity Formula:
<pre>
Similarity = (A · B) / (||A|| ||B||)
</pre>


Value close to 1 → Same person

Value close to 0 → Different person

Threshold (default = 0.7):

0.7 → Recognized

< 0.7 → Not recognized
<pre>
🧠 Data Flow Summary
Registration Flow
Camera → MTCNN → FaceNet → Embedding → JSON Storage
                                  ↓
                         StudentDetails CSV

Attendance Flow
Camera → MTCNN → FaceNet → Embedding
                                ↓
                    Compare with Stored Embeddings
                                ↓
                      If Match → Mark Attendance
                                ↓
                         Save to CSV File
</pre>
⚙️ How To Run The Project

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Start Server

uvicorn p3:app --reload

3️⃣ Open Swagger UI

http://localhost:8000/docs

📂 Data Persistence Strategy

Data Type	Storage Format	Location

Face Embeddings	JSON	FaceEmbeddings/

Student Info	CSV	StudentDetails/

Attendance	CSV	Attendance/

🔐 Why JSON for Embeddings?

Easy to serialize

Lightweight

Human-readable

Fast load/save operations

📈 Performance Notes

GPU supported (CUDA if available)

Default threshold = 0.7

Accuracy improves with more registration samples

Real-time processing
