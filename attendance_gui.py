import cv2
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# ----------------- Setup -----------------
attendance_file = "Attendance.csv"
images_folder = "Images"
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
known_faces = {}  # "rollno_name": [face_images]
marked_today = set()

# ----------------- Load Faces -----------------
def load_known_faces():
    global known_faces
    known_faces = {}
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            try:
                rollno, name, _ = os.path.splitext(filename)[0].split("_")
            except ValueError:
                continue
            img = cv2.imread(os.path.join(images_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                key = f"{rollno}_{name}"
                if key not in known_faces:
                    known_faces[key] = []
                known_faces[key].append(img)

load_known_faces()

# ----------------- Attendance -----------------
def markAttendance(name, rollno):
    global marked_today
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
        except:
            df = pd.DataFrame(columns=["Roll No","Name","Date","Time"])
    else:
        df = pd.DataFrame(columns=["Roll No","Name","Date","Time"])
    
    if rollno not in marked_today:
        df = pd.concat([df, pd.DataFrame([[rollno,name,date,time]], columns=df.columns)], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        marked_today.add(rollno)

# ----------------- Tkinter GUI -----------------
root = tk.Tk()
root.title("Face Attendance Dashboard")
root.geometry("1150x750")
root.resizable(False, False)

ttk.Label(root, text="Face Attendance Dashboard", font=("Helvetica", 22, "bold")).pack(pady=10)

dashboard_frame = ttk.Frame(root)
dashboard_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Attendance list
attendance_frame = ttk.LabelFrame(dashboard_frame, text="Today's Attendance", padding=10)
attendance_frame.grid(row=0, column=0, sticky="ns", padx=10)
attendance_listbox = tk.Listbox(attendance_frame, width=35, height=25)
attendance_listbox.pack()
attendance_count_label = ttk.Label(attendance_frame, text="Total Present Today: 0", font=("Helvetica", 12, "bold"))
attendance_count_label.pack(pady=5)

# Camera feed
camera_frame = ttk.LabelFrame(dashboard_frame, text="Camera Feed", padding=10)
camera_frame.grid(row=0, column=1, padx=10)
canvas = tk.Canvas(camera_frame, width=640, height=480)
canvas.pack()
canvas_img_id = canvas.create_image(0, 0, anchor=tk.NW, image=None)

# Buttons
button_frame = ttk.LabelFrame(dashboard_frame, text="Controls", padding=10)
button_frame.grid(row=0, column=2, sticky="n", padx=10)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
running = False
frame_imgtk = None

# ----------------- GUI Functions -----------------
def start_attendance():
    global running, marked_today
    running = True
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
            marked_today = set(df[df["Date"] == today]["Roll No"].tolist())
        except:
            marked_today = set()
    update_frame()

def stop_attendance():
    global running
    running = False

def exit_program():
    global running
    running = False
    cap.release()
    root.destroy()

def register_new_person():
    global running
    running = False
    name = simpledialog.askstring("Register", "Enter name:")
    rollno = simpledialog.askstring("Register", "Enter roll number:")
    if not name or not rollno:
        start_attendance()
        return

    reg_window = tk.Toplevel(root)
    reg_window.title(f"Register {name} ({rollno})")
    reg_canvas = tk.Canvas(reg_window, width=640, height=480)
    reg_canvas.pack()
    reg_canvas.img_id = reg_canvas.create_image(0, 0, anchor=tk.NW, image=None)
    img_count = [0]

    def capture_image():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (100,100))
                img_count[0] +=1
                filepath = os.path.join(images_folder,f"{rollno}_{name}_{img_count[0]}.jpg")
                cv2.imwrite(filepath, face_resized)
            if img_count[0]>=5:
                messagebox.showinfo("Success", f"Captured {img_count[0]} images for {name}")
                load_known_faces()
                reg_window.destroy()
                start_attendance()

    def update_reg_frame():
        if reg_window.winfo_exists():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).resize((640,480))
                imgtk = ImageTk.PhotoImage(image=img)
                reg_canvas.imgtk = imgtk
                reg_canvas.itemconfig(reg_canvas.img_id, image=imgtk)
            reg_window.after(10, update_reg_frame)

    ttk.Button(reg_window, text="Capture Images", command=capture_image).pack(pady=5)
    update_reg_frame()

def match_face(face_img):
    face_resized = cv2.resize(face_img, (100,100))
    for key, faces in known_faces.items():
        for stored_face in faces:
            res = cv2.matchTemplate(face_resized, stored_face, cv2.TM_CCOEFF_NORMED)
            if res >= 0.7:  # threshold
                rollno, name = key.split("_")
                return rollno, name
    return "-", "Unknown"

def update_frame():
    global frame_imgtk
    if running:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
            for (x,y,w,h) in faces:
                face_img = gray[y:y+h, x:x+w]
                rollno, name = match_face(face_img)
                if rollno!="-":
                    markAttendance(name, rollno)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"{rollno} {name}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            # Update attendance list
            try:
                df = pd.read_csv(attendance_file)
                today = datetime.now().strftime("%Y-%m-%d")
                present_today = df[df["Date"]==today][["Roll No","Name"]].values.tolist()
                attendance_listbox.delete(0, tk.END)
                for r,n in present_today:
                    attendance_listbox.insert(tk.END,f"{r} - {n}")
                attendance_count_label.config(text=f"Total Present Today: {len(present_today)}")
            except:
                pass

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((640,480))
            frame_imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = frame_imgtk
            canvas.itemconfig(canvas_img_id,image=frame_imgtk)

        root.after(10, update_frame)

# ----------------- Start -----------------
start_attendance()
root.mainloop()
