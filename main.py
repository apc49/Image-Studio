import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ExifTags
import numpy as np
import shutil
import json
import sqlite3
import threading
import subprocess

# Try to import ttkbootstrap
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    ttk_bootstrap = True
except ImportError:
    print("ttkbootstrap not installed. Using standard ttk.")
    from tkinter import ttk
    ttk_bootstrap = False
    # Define constants that would be in ttkbootstrap
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    INVERSE = "inverse"

# Try to import tkinterdnd2 for drag-and-drop
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    drag_drop_support = True
except ImportError:
    print("tkinterdnd2 not installed. Drag-and-drop disabled.")
    drag_drop_support = False
    # Mock TkinterDnD.Tk with a standard tk.Tk
    class TkinterDndMock:
        @staticmethod
        def Tk():
            return tk.Tk()
    TkinterDnD = TkinterDndMock

# Try optional system monitoring
try:
    import psutil
    system_monitoring = True
except ImportError:
    print("psutil not installed. System monitoring disabled.")
    system_monitoring = False

# Try to import docx
try:
    from docx import Document
    docx_support = True
except ImportError:
    print("python-docx not installed. DOCX support disabled.")
    docx_support = False

# Try to import optional AI libraries
try:
    import torch
    torch_available = True
    DEVICE = torch.device("cpu")
except ImportError:
    print("PyTorch not installed. AI features disabled.")
    torch_available = False
    DEVICE = None

# Only import these if torch is available
if torch_available:
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        transformers_available = True
    except ImportError:
        print("Transformers not installed. Image captioning disabled.")
        transformers_available = False
    
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        print("Ultralytics not installed. Object detection disabled.")
        yolo_available = False

# Try to import OCR and image hashing
try:
    import pytesseract
    ocr_available = True
except ImportError:
    print("Pytesseract not installed. OCR disabled.")
    ocr_available = False

try:
    import imagehash
    imagehash_available = True
except ImportError:
    print("Imagehash not installed. Image hashing disabled.")
    imagehash_available = False

# Try to import OpenCV
try:
    import cv2
    opencv_available = True
except ImportError:
    print("OpenCV not installed. Image processing features limited.")
    opencv_available = False

# Try to import face_recognition
try:
    import face_recognition
    face_recognition_available = True
except ImportError:
    print("Face recognition not installed. Face detection disabled.")
    face_recognition_available = None

########################################################################
#                        GLOBAL CONSTANTS
########################################################################
MAX_DISPLAY_WIDTH = 700
MAX_DISPLAY_HEIGHT = 700
DB_FILE = "image_data.db"  # SQLite database file
DOCX_FILE = "Business - Charector Prompts.docx"  # DOCX file for prepopulation
GALLERY_FOLDER = "gallery"  # Folder to store gallery images
SPLASH_IMAGE = "IMG_3735.JPG"  # Splash screen image

if not os.path.exists(GALLERY_FOLDER):
    os.makedirs(GALLERY_FOLDER)

current_file_path = None      # For image analysis page
training_folder = None        # For training interface
current_theme = "superhero" if ttk_bootstrap else "default"   # Default theme
resizing_in_progress = False  # Guard flag for resize events

# Global list for process logs
process_logs = []

# Global variable to hold the splash image (if used)
splash_photo = None

# Global variables for UI elements
image_label = None
analysis_text = None
gallery_listbox = None
gallery_preview = None
details_text = None
process_log_text = None
pip_tree = None
prompt_preview = None
analysis_image_path = None
analysis_result_widget = None

# AI models (initialized later if dependencies are available)
processor = None
blip_model = None
yolo_model = None

########################################################################
#                   DATABASE INITIALIZATION & FUNCTIONS
########################################################################
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    file_path TEXT,
    caption TEXT,
    objects TEXT,
    face TEXT,
    ocr TEXT,
    image_hash TEXT,
    exif TEXT,
    top_colors TEXT,
    prompt TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_text TEXT,
    options TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS gallery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    file_name TEXT,
    title TEXT,
    description TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def insert_prompt(prompt_text, options_dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO prompts (prompt_text, options) VALUES (?, ?)",
    (prompt_text, json.dumps(options_dict)))
    conn.commit()
    conn.close()

def gallery_insert_image(file_path, file_name, title, description):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO gallery (file_path, file_name, title, description) VALUES (?, ?, ?, ?)",
    (file_path, file_name, title, description))
    conn.commit()
    conn.close()

def gallery_get_images():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, file_path, file_name, title, description FROM gallery ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def gallery_delete_image_db(image_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM gallery WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()

def gallery_update_image(image_id, title, description):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE gallery SET title = ?, description = ? WHERE id = ?", (title, description, image_id))
    conn.commit()
    conn.close()

########################################################################
#                HELPER FUNCTIONS (Image Analysis)
########################################################################
def scale_pil_image(pil_img, max_w, max_h):
    w, h = pil_img.size
    ratio = min(max_w / w, max_h / h)
    if ratio < 1.0:
        new_size = (int(w * ratio), int(h * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    return pil_img

def generate_caption(image_path):
    if not transformers_available:
        return "Image captioning not available (transformers library missing)"
    
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = blip_model.generate(**inputs)
            return processor.decode(out_ids[0], skip_special_tokens=True)
    except Exception as e:
        append_log(f"Caption generation error: {e}")
        return f"Caption error: {str(e)}"

def detect_objects(image_path):
    if not yolo_available:
        return []
    
    try:
        results = yolo_model(image_path)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                conf = float(box.conf[0])
                detections.append((label, conf))
        return detections
    except Exception as e:
        append_log(f"Object detection error: {e}")
        return []

def detect_and_draw(image_path):
    if not opencv_available or not yolo_available:
        # Just return the original image if OpenCV or YOLO is not available
        img = np.array(Image.open(image_path).convert("RGB"))
        return img
    
    try:
        results = yolo_model(image_path)
        img = cv2.imread(image_path)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                conf = float(box.conf[0])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img
    except Exception as e:
        append_log(f"Detect and draw error: {e}")
        img = np.array(Image.open(image_path).convert("RGB"))
        return img

def get_exif_data(image_path):
    exif = {}
    try:
        pil_img = Image.open(image_path)
        raw = pil_img._getexif()
        if raw:
            for tag, value in raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif[tag_name] = value
    except Exception as e:
        append_log(f"EXIF error: {e}")
    return exif

def analyze_colors(image_path, top_n=3):
    try:
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize((200, 200))
        arr = np.array(pil_img).reshape(-1, 3)
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        sorted_idxs = np.argsort(-counts)
        return [tuple(unique[idx]) for idx in sorted_idxs[:top_n]]
    except Exception as e:
        append_log(f"Color analysis error: {e}")
        return []

def do_ocr(image_path):
    if not ocr_available:
        return "OCR not available (pytesseract library missing)"
    
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        append_log(f"OCR error: {e}")
        return f"OCR error: {str(e)}"

def face_detect(image_path):
    if face_recognition is None:
        return "Face detection not available (face_recognition library missing)"
    
    try:
        img = face_recognition.load_image_file(image_path)
        locs = face_recognition.face_locations(img)
        return f"{len(locs)} face(s) detected"
    except Exception as e:
        append_log(f"Face error: {e}")
        return f"Face error: {e}"

def image_hash_value(image_path):
    if not imagehash_available:
        return "Image hashing not available (imagehash library missing)"
    
    try:
        pil_img = Image.open(image_path).convert("RGB")
        ph = imagehash.phash(pil_img)
        return str(ph)
    except Exception as e:
        append_log(f"Imagehash error: {e}")
        return f"Imagehash error: {str(e)}"

def select_image_for_analysis():
    filetypes = [("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    filepath = filedialog.askopenfilename(title="Select Image for Analysis", filetypes=filetypes)
    if filepath:
        global analysis_image_path
        analysis_image_path = filepath
        analysis_result_widget.insert(tk.END, f"\n📁 Selected: {filepath}\n")

def run_analysis():
    try:
        if not analysis_image_path:
            analysis_result_widget.insert(tk.END, "\n⚠️ No image selected.\n")
            return

        analysis_result_widget.insert(tk.END, "\n🔍 Analyzing image...\n")
        # Example placeholder output
        analysis_result_widget.insert(tk.END, "✅ Analysis complete. (Note: Full AI analysis disabled due to missing dependencies)\n")
    except Exception as e:
        analysis_result_widget.insert(tk.END, f"❌ Error: {str(e)}\n")

########################################################################
#               PROMPT GENERATOR (Image Analysis)
########################################################################
def generate_prompt_str(caption, objects, face, ocr_text, exif, top_colors, width, height):
    obj_str = ", ".join([f"{lbl}({conf:.2f})" for lbl, conf in objects]) if objects else "none"
    aspect_ratio = round(width / height, 2) if height else "N/A"
    camera = f"{exif.get('Make', 'N/A')} {exif.get('Model', 'N/A')}".strip()
    color_str = ", ".join([f"RGB{col}" for col in top_colors]) if top_colors else "No color data"
    prompt = (
    f"Caption: {caption}\n"
    f"Objects: {obj_str}\n"
    f"Faces: {face}\n"
    f"OCR: {ocr_text.strip()[:100]}...\n"
    f"Camera: {camera}\n"
    f"Resolution: {width}x{height} (Aspect {aspect_ratio})\n"
    f"Top Colors: {color_str}"
    )
    return prompt

########################################################################
#                     MODEL LOADING (Image Analysis)
########################################################################
def load_models():
    global processor, blip_model, yolo_model
    
    if torch_available and transformers_available:
        print("Loading BLIP model (CPU)...")
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
            blip_model.eval()
            append_log("BLIP model loaded successfully")
        except Exception as e:
            append_log(f"Error loading BLIP model: {e}")
    
    if torch_available and yolo_available:
        print("Loading YOLOv8 model (CPU)...")
        try:
            yolo_model = YOLO("yolov8n.pt")
            append_log("YOLO model loaded successfully")
        except Exception as e:
            append_log(f"Error loading YOLO model: {e}")

########################################################################
#           PROCESS LOGGING FUNCTIONS & PAGE
########################################################################
def append_log(message):
    global process_logs, process_log_text
    log_entry = f"{message}\n"
    process_logs.append(log_entry)
    if process_log_text is not None:
        process_log_text.insert(tk.END, log_entry)
        process_log_text.see(tk.END)
    print(log_entry.strip())

def update_pip_installs():
    global pip_tree
    if pip_tree is None:
        return
    
    try:
        all_proc = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True, check=True)
        all_packages = json.loads(all_proc.stdout)
    except Exception as e:
        all_packages = []
        append_log(f"Error getting pip list: {e}")
    
    try:
        outdated_proc = subprocess.run(["pip", "list", "--outdated", "--format=json"], capture_output=True, text=True, check=True)
        outdated_packages = json.loads(outdated_proc.stdout)
        outdated_dict = {pkg["name"].lower(): pkg for pkg in outdated_packages}
    except Exception as e:
        outdated_dict = {}
        append_log(f"Error getting pip outdated: {e}")
    
    for item in pip_tree.get_children():
        pip_tree.delete(item)
    
    for pkg in all_packages:
        name = pkg["name"]
        current_ver = pkg["version"]
        if name.lower() in outdated_dict:
            latest = outdated_dict[name.lower()]["latest_version"]
            color = "red"
        else:
            latest = current_ver
            color = "green"
        pip_tree.insert("", "end", values=(name, current_ver, latest), tags=(color,))
    
    pip_tree.tag_configure("red", foreground="red")
    pip_tree.tag_configure("green", foreground="green")
    
    # Reschedule next update after 60 seconds
    if process_log_text:
        process_log_text.after(60000, update_pip_installs)

def build_process_log_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    # Create a frame to hold 3 columns: left for process log, right for pip info
    log_frame = ttk.Frame(frame)
    log_frame.grid(row=0, column=0, sticky="nsew")
    log_frame.columnconfigure(0, weight=1)
    log_frame.columnconfigure(1, weight=1)
    log_frame.columnconfigure(2, weight=1)

    # Process log text (column 0)
    ttk.Label(log_frame, text="Process Log", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    global process_log_text
    process_log_text = tk.Text(log_frame, wrap=tk.WORD)
    process_log_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    # Display already logged messages
    for log_entry in process_logs:
        process_log_text.insert(tk.END, log_entry)

    # (Optional) Middle column can be used for additional info – here we leave it blank
    ttk.Label(log_frame, text="").grid(row=0, column=1)

    # Pip packages info (column 2) using a Treeview
    ttk.Label(log_frame, text="Pip Packages", font=("Helvetica", 16)).grid(row=0, column=2, pady=10)
    global pip_tree
    pip_tree = ttk.Treeview(log_frame, columns=("Name", "Current", "Latest"), show="headings", height=10)
    pip_tree.heading("Name", text="Name")
    pip_tree.heading("Current", text="Current")
    pip_tree.heading("Latest", text="Latest")
    pip_tree.column("Name", anchor="w", width=150)
    pip_tree.column("Current", anchor="center", width=80)
    pip_tree.column("Latest", anchor="center", width=80)
    pip_tree.grid(row=1, column=2, padx=10, pady=5, sticky="nsew")
    update_pip_installs()

    ttk.Button(log_frame, text="Clear Log", command=lambda: process_log_text.delete("1.0", tk.END)).grid(row=2, column=0, pady=10)
    ttk.Button(log_frame, text="Back to Home", command=show_splash_frame).grid(row=2, column=2, pady=10)

########################################################################
#        SPLASH SCREEN (Home) & THEME TOGGLE
########################################################################
def build_splash_frame(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    # For now, disable splash image and use fallback text
    fallback_label = ttk.Label(frame, text="AWDigitalworld", font=("Helvetica", 24, "bold"))
    fallback_label.grid(row=0, column=0, sticky="nsew")
    subtitle_label = ttk.Label(frame, text="Forever Starts Here", font=("Helvetica", 18))
    subtitle_label.grid(row=1, column=0, pady=10)
    
    # Show dependency status
    status_frame = ttk.Frame(frame)
    status_frame.grid(row=2, column=0, pady=20, sticky="nsew")
    
    ttk.Label(status_frame, text="Dependency Status:", font=("Helvetica", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
    
    row = 1
    for name, status, desc in [
        ("ttkbootstrap", ttk_bootstrap, "Enhanced UI themes"),
        ("tkinterdnd2", drag_drop_support, "Drag and drop support"),
        ("OpenCV (cv2)", opencv_available, "Image processing"),
        ("PyTorch", torch_available, "AI model support"),
        ("Transformers", transformers_available, "Image captioning"),
        ("YOLO", yolo_available, "Object detection"),
        ("pytesseract", ocr_available, "Text recognition (OCR)"),
        ("face_recognition", face_recognition_available, "Face detection"),
        ("imagehash", imagehash_available, "Image hashing"),
        ("psutil", system_monitoring, "System monitoring"),
        ("python-docx", docx_support, "Document support")
    ]:
        status_text = "✅ Available" if status else "❌ Missing"
        status_color = "green" if status else "red"
        ttk.Label(status_frame, text=f"{name}: ").grid(row=row, column=0, sticky="w", padx=20)
        status_label = ttk.Label(status_frame, text=status_text, foreground=status_color)
        status_label.grid(row=row, column=1, sticky="w")
        ttk.Label(status_frame, text=f"({desc})").grid(row=row, column=2, sticky="w", padx=10)
        row += 1
    
    # Installation instructions
    ttk.Label(status_frame, text="To enable all features, install missing dependencies:", font=("Helvetica", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(20, 5))
    row += 1
    ttk.Label(status_frame, text="pip install -r requirements.txt").grid(row=row, column=0, columnspan=3, sticky="w", padx=20)

def toggle_theme():
    global current_theme
    if not ttk_bootstrap:
        messagebox.showinfo("Theme Toggle", "Theme toggle requires ttkbootstrap library")
        return
    
    try:
        available_themes = style.theme_names()
        if current_theme == "superhero" and "darkly" in available_themes:
            style.theme_use("darkly")
            current_theme = "darkly"
            append_log("Theme changed to darkly")
        elif current_theme == "darkly" and "superhero" in available_themes:
            style.theme_use("superhero")
            current_theme = "superhero"
            append_log("Theme changed to superhero")
        else:
            messagebox.showwarning("Theme Error", "Desired theme not available.")
    except Exception as e:
        append_log(f"Theme toggle error: {e}")

########################################################################
#       FRAME SWITCHING FUNCTIONS
########################################################################
def show_splash_frame():
    splash_frame.tkraise()

def show_main_frame():
    analysis_frame.tkraise()

def show_prompt_frame():
    prompt_frame.tkraise()

def show_training_frame():
    training_frame.tkraise()

def show_gallery_frame():
    gallery_frame.tkraise()
    load_gallery_list()

def show_process_log_frame():
    process_log_frame.tkraise()

def show_only(frame):
    frame.tkraise()

########################################################################
#           IMAGE ANALYSIS: on_select_image Function
########################################################################
def on_select_image():
    global tk_image, current_file_path
    file_path = filedialog.askopenfilename(
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
    )
    if not file_path:
        return
    app.after(100, lambda: on_select_image_from_file(file_path))

def on_select_image_from_file(file_path):
    global tk_image, current_file_path
    if not os.path.exists(file_path):
        messagebox.showerror("File Error", f"File does not exist: {file_path}")
        append_log(f"File not found: {file_path}")
        return
    try:
        current_file_path = file_path
        if opencv_available and yolo_available:
            cv_img = detect_and_draw(file_path)
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img_rgb)
        else:
            pil_img = Image.open(file_path).convert("RGB")
        
        pil_img = scale_pil_image(pil_img, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
        tk_image = ImageTk.PhotoImage(pil_img)
        image_label.config(image=tk_image)
        
        # Get image analysis data based on available dependencies
        caption = generate_caption(file_path) if transformers_available else "Image captioning not available"
        objects = detect_objects(file_path) if yolo_available else []
        exif = get_exif_data(file_path)
        top_colors = analyze_colors(file_path)
        ocr_text = do_ocr(file_path) if ocr_available else "OCR not available"
        face_result = face_detect(file_path) if face_recognition_available else "Face detection not available"
        
        with Image.open(file_path) as im:
            width, height = im.size
            prompt_str = generate_prompt_str(caption, objects, face_result, ocr_text, exif, top_colors, width, height)
            analysis_text.delete("1.0", tk.END)
            analysis_text.insert(tk.END, prompt_str)
            append_log(f"Processed image: {file_path}")
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error processing image: {e}")
        append_log(f"Processing error: {e}")

def drop_event(event):
    if drag_drop_support:
        files = app.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            app.after(0, lambda: on_select_image_from_file(file_path))

########################################################################
#         RESIZE HANDLER WITH GUARD
########################################################################
def resize_handler(event):
    global resizing_in_progress, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
    if event.widget == app and not resizing_in_progress:
        resizing_in_progress = True
        MAX_DISPLAY_WIDTH = app.winfo_width() - 400
        MAX_DISPLAY_HEIGHT = app.winfo_height() - 100
        if current_file_path and os.path.exists(current_file_path):
            try:
                on_select_image_from_file(current_file_path)
            except Exception as e:
                append_log(f"Resize processing error: {e}")
        app.after(200, lambda: set_resizing_flag(False))

def set_resizing_flag(value):
    global resizing_in_progress
    resizing_in_progress = value

########################################################################
#         IMAGE ANALYSIS: Other Functions
########################################################################
def copy_text():
    txt = analysis_text.get("1.0", tk.END)
    app.clipboard_clear()
    app.clipboard_append(txt)
    messagebox.showinfo("Copied", "Prompt copied to clipboard!")

def save_prompt():
    txt = analysis_text.get("1.0", tk.END)
    if not txt.strip():
        messagebox.showwarning("Save Prompt", "There is no prompt to save!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt)
        messagebox.showinfo("Save Prompt", "Prompt saved successfully!")
        append_log(f"Prompt saved to {file_path}")

def regenerate_prompt():
    global current_file_path
    if not current_file_path:
        messagebox.showwarning("Regenerate", "No image is loaded!")
        return
    on_select_image_from_file(current_file_path)

def clear_prompt():
    analysis_text.delete("1.0", tk.END)

########################################################################
#         TRAINING FUNCTIONS (Simple Classifier Training)
########################################################################
def train_model(training_dir, learning_rate, epochs, batch_size, log_callback):
    if not torch_available:
        log_callback("Training requires PyTorch which is not installed.\n")
        append_log("Training failed: PyTorch not available")
        return
    
    try:
        from torch.utils.data import DataLoader
        from torchvision import models, datasets, transforms
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        log_callback("Training requires torchvision which is not installed.\n")
        append_log("Training failed: torchvision not available")
        return
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        train_dataset = datasets.ImageFolder(training_dir, transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        log_callback(f"Error loading dataset: {e}\n")
        append_log(f"Training dataset error: {e}")
        return
    
    num_classes = len(train_dataset.classes)
    log_callback(f"Found {len(train_dataset)} images in {num_classes} classes.\n")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    log_callback("Starting training...\n")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        log_callback(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}\n")
    
    log_callback("Training complete.\n")
    
    # Save the model
    try:
        if not os.path.exists("models"):
            os.makedirs("models")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/model_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        log_callback(f"Model saved to {model_path}\n")
        append_log(f"Model saved to {model_path}")
    except Exception as e:
        log_callback(f"Error saving model: {e}\n")
        append_log(f"Error saving model: {e}")

########################################################################
#           IMAGE ANALYSIS INTERFACE
########################################################################
def build_image_analysis(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    
    # Create a title label
    ttk.Label(frame, text="Image Analysis", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    # Create a container for the image analysis interface
    analysis_container = ttk.Frame(frame)
    analysis_container.grid(row=1, column=0, sticky="nsew")
    analysis_container.columnconfigure(0, weight=2)
    analysis_container.columnconfigure(1, weight=1)
    analysis_container.rowconfigure(0, weight=1)
    
    # Left side: Image display
    img_frame = ttk.Frame(analysis_container)
    img_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    global image_label
    image_label = ttk.Label(img_frame)
    image_label.pack(expand=True, fill="both")
    
    # Right side: Analysis results and controls
    control_frame = ttk.Frame(analysis_container)
    control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    control_frame.columnconfigure(0, weight=1)
    control_frame.rowconfigure(1, weight=1)
    
    # Analysis text area
    ttk.Label(control_frame, text="Analysis Results", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
    
    global analysis_text
    analysis_text = tk.Text(control_frame, wrap=tk.WORD, height=15)
    analysis_text.grid(row=1, column=0, sticky="nsew", pady=5)
    analysis_scroll = ttk.Scrollbar(control_frame, orient="vertical", command=analysis_text.yview)
    analysis_scroll.grid(row=1, column=1, sticky="ns", pady=5)
    analysis_text.config(yscrollcommand=analysis_scroll.set)
    
    # Control buttons
    btn_frame = ttk.Frame(control_frame)
    btn_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
    
    ttk.Button(btn_frame, text="Select Image", command=on_select_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Copy Text", command=copy_text).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Save Prompt", command=save_prompt).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Regenerate", command=regenerate_prompt).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Clear", command=clear_prompt).pack(side=tk.LEFT, padx=5)
    
    # Initialize drag and drop
    if drag_drop_support:
        image_label.drop_target_register(DND_FILES)
        image_label.dnd_bind('<<Drop>>', drop_event)

########################################################################
#         TRAINING INTERFACE
########################################################################
def build_training_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    
    # Create a title label
    ttk.Label(frame, text="AI Model Training", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    if not torch_available:
        ttk.Label(frame, text="PyTorch is not installed. Model training is not available.", 
                 font=("Helvetica", 12), foreground="red").grid(row=1, column=0, pady=20)
        ttk.Button(frame, text="Back to Home", command=show_splash_frame).grid(row=2, column=0, pady=10)
        return
    
    # Create a container for the training interface
    training_container = ttk.Frame(frame)
    training_container.grid(row=1, column=0, sticky="nsew")
    training_container.columnconfigure(0, weight=1)
    training_container.columnconfigure(1, weight=1)
    training_container.rowconfigure(1, weight=1)
    
    # Left side: Training parameters
    param_frame = ttk.Frame(training_container)
    param_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    param_frame.columnconfigure(1, weight=1)
    
    # Model configuration
    ttk.Label(param_frame, text="Training Configuration", font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
    
    # Data folder
    ttk.Label(param_frame, text="Data Folder:").grid(row=1, column=0, sticky="w", pady=5)
    folder_frame = ttk.Frame(param_frame)
    folder_frame.grid(row=1, column=1, sticky="ew", pady=5)
    folder_var = tk.StringVar()
    folder_entry = ttk.Entry(folder_frame, textvariable=folder_var, state="readonly")
    folder_entry.pack(side=tk.LEFT, fill="x", expand=True)
    
    def select_training_folder():
        global training_folder
        folder = filedialog.askdirectory(title="Select Training Data Folder")
        if folder:
            training_folder = folder
            folder_var.set(folder)
    
    ttk.Button(folder_frame, text="Browse", command=select_training_folder).pack(side=tk.RIGHT, padx=5)
    
    # Hyperparameters
    ttk.Label(param_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", pady=5)
    lr_var = tk.DoubleVar(value=0.001)
    ttk.Entry(param_frame, textvariable=lr_var).grid(row=2, column=1, sticky="ew", pady=5)
    
    ttk.Label(param_frame, text="Epochs:").grid(row=3, column=0, sticky="w", pady=5)
    epochs_var = tk.IntVar(value=5)
    ttk.Entry(param_frame, textvariable=epochs_var).grid(row=3, column=1, sticky="ew", pady=5)
    
    ttk.Label(param_frame, text="Batch Size:").grid(row=4, column=0, sticky="w", pady=5)
    batch_var = tk.IntVar(value=16)
    ttk.Entry(param_frame, textvariable=batch_var).grid(row=4, column=1, sticky="ew", pady=5)
    
    # Right side: Training log
    log_frame = ttk.Frame(training_container)
    log_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(1, weight=1)
    
    ttk.Label(log_frame, text="Training Log", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
    
    training_log = tk.Text(log_frame, wrap=tk.WORD)
    training_log.grid(row=1, column=0, sticky="nsew")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=training_log.yview)
    log_scroll.grid(row=1, column=1, sticky="ns")
    training_log.config(yscrollcommand=log_scroll.set)
    
    # Control buttons
    btn_frame = ttk.Frame(training_container)
    btn_frame.grid(row=1, column=0, pady=10, sticky="s")
    
    def start_training():
        if not training_folder:
            messagebox.showwarning("Training Error", "Please select a training data folder.")
            return
        
        lr = lr_var.get()
        epochs = epochs_var.get()
        batch_size = batch_var.get()
        
        # Clear the log
        training_log.delete("1.0", tk.END)
        
        # Log the training configuration
        training_log.insert(tk.END, f"Starting training with configuration:\n")
        training_log.insert(tk.END, f"Folder: {training_folder}\n")
        training_log.insert(tk.END, f"Learning Rate: {lr}\n")
        training_log.insert(tk.END, f"Epochs: {epochs}\n")
        training_log.insert(tk.END, f"Batch Size: {batch_size}\n\n")
        
        # Start training in a separate thread
        def train_thread():
            train_model(training_folder, lr, epochs, batch_size, 
                        lambda msg: app.after(0, lambda: training_log.insert(tk.END, msg)))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    ttk.Button(btn_frame, text="Start Training", command=start_training).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Cancel", command=lambda: training_log.insert(tk.END, "Training canceled by user\n")).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Back to Home", command=show_splash_frame).pack(side=tk.LEFT, padx=5)

########################################################################
#         GALLERY INTERFACE
########################################################################
def build_gallery_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    
    # Create a title label
    ttk.Label(frame, text="Image Gallery", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    # Create a container for the gallery interface
    gallery_container = ttk.Frame(frame)
    gallery_container.grid(row=1, column=0, sticky="nsew")
    gallery_container.columnconfigure(0, weight=1)
    gallery_container.columnconfigure(1, weight=2)
    gallery_container.rowconfigure(0, weight=1)
    
    # Left side: Gallery listing
    list_frame = ttk.Frame(gallery_container)
    list_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(1, weight=1)
    
    ttk.Label(list_frame, text="Gallery Images", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
    
    global gallery_listbox
    gallery_listbox = tk.Listbox(list_frame)
    gallery_listbox.grid(row=1, column=0, sticky="nsew")
    gallery_listbox.bind('<<ListboxSelect>>', on_gallery_select)
    
    list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=gallery_listbox.yview)
    list_scroll.grid(row=1, column=1, sticky="ns")
    gallery_listbox.config(yscrollcommand=list_scroll.set)
    
    # Control buttons for the list
    list_btn_frame = ttk.Frame(list_frame)
    list_btn_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
    
    ttk.Button(list_btn_frame, text="Upload Image", command=gallery_upload_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(list_btn_frame, text="Download Image", command=gallery_download_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(list_btn_frame, text="Delete Image", command=gallery_delete_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(list_btn_frame, text="Edit Details", command=gallery_edit_image).pack(side=tk.LEFT, padx=5)
    
    # Right side: Image preview and details
    preview_frame = ttk.Frame(gallery_container)
    preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)
    preview_frame.rowconfigure(1, weight=1)
    
    # Preview image
    preview_container = ttk.Frame(preview_frame)
    preview_container.grid(row=0, column=0, sticky="nsew", pady=5)
    preview_container.columnconfigure(0, weight=1)
    preview_container.rowconfigure(0, weight=1)
    
    global gallery_preview, gallery_preview_photo
    gallery_preview = ttk.Label(preview_container, text="No Image Selected")
    gallery_preview.grid(row=0, column=0, sticky="nsew")
    gallery_preview_photo = None
    
    # Image details
    details_container = ttk.Frame(preview_frame)
    details_container.grid(row=1, column=0, sticky="nsew", pady=5)
    details_container.columnconfigure(0, weight=1)
    details_container.rowconfigure(1, weight=1)
    
    ttk.Label(details_container, text="Image Details", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
    
    global details_text
    details_text = tk.Text(details_container, wrap=tk.WORD, height=6)
    details_text.grid(row=1, column=0, sticky="nsew")
    details_scroll = ttk.Scrollbar(details_container, orient="vertical", command=details_text.yview)
    details_scroll.grid(row=1, column=1, sticky="ns")
    details_text.config(yscrollcommand=details_scroll.set)
    
    # Navigation button
    ttk.Button(preview_frame, text="Back to Home", command=show_splash_frame).grid(row=2, column=0, pady=10)
    
    # Load gallery images
    load_gallery_list()

########################################################################
#         PROMPT SUGGESTIONS INTERFACE
########################################################################
def build_prompt_suggestions_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    
    # Create a title label
    ttk.Label(frame, text="Prompt Suggestions", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    # Create a container for the suggestion interface
    suggestion_container = ttk.Frame(frame)
    suggestion_container.grid(row=1, column=0, sticky="nsew")
    suggestion_container.columnconfigure(0, weight=1)
    suggestion_container.rowconfigure(1, weight=1)
    
    # Suggestion generation controls
    control_frame = ttk.Frame(suggestion_container)
    control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    
    ttk.Label(control_frame, text="Generate prompts based on:").pack(side=tk.LEFT, padx=5)
    
    theme_var = tk.StringVar(value="Portrait")
    theme_combo = ttk.Combobox(control_frame, textvariable=theme_var, values=["Portrait", "Landscape", "Product", "Conceptual", "Abstract", "Documentary"])
    theme_combo.pack(side=tk.LEFT, padx=5)
    
    ttk.Button(control_frame, text="Generate Suggestions", command=lambda: generate_suggestions(theme_var.get(), suggestion_text)).pack(side=tk.LEFT, padx=5)
    
    # Suggestions text area
    suggestion_text = tk.Text(suggestion_container, wrap=tk.WORD)
    suggestion_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    # Button frame
    btn_frame = ttk.Frame(suggestion_container)
    btn_frame.grid(row=2, column=0, pady=10)
    
    ttk.Button(btn_frame, text="Copy to Clipboard", command=lambda: copy_to_clipboard(suggestion_text)).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Save Suggestions", command=lambda: save_suggestions(suggestion_text)).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Back to Home", command=show_splash_frame).pack(side=tk.LEFT, padx=5)

def generate_suggestions(theme, text_widget):
    """Generate prompt suggestions based on the theme."""
    text_widget.delete("1.0", tk.END)
    
    suggestions = {
        "Portrait": [
            "A professional headshot with natural lighting and neutral background",
            "An environmental portrait showing personality and profession",
            "A moody portrait with dramatic side lighting and deep shadows",
            "A candid moment with genuine emotion and soft bokeh background",
            "A stylized portrait with vivid colors and creative composition"
        ],
        "Landscape": [
            "A serene sunrise over mountains with fog in the valley",
            "A dramatic storm approaching across a vast plain",
            "A golden hour shot of rolling hills with long shadows",
            "A night landscape with stars and silhouetted elements",
            "An intimate landscape focusing on patterns in nature"
        ],
        "Product": [
            "A clean product shot on white background with soft shadows",
            "A lifestyle product image showing the item in use",
            "A dramatic product image with creative lighting and reflections",
            "A flatlay arrangement with complementary props and textures",
            "A close-up detail shot highlighting craftsmanship"
        ],
        "Conceptual": [
            "An abstract representation of time using motion blur",
            "A surreal composite blending reality and imagination",
            "A minimalist concept with strong geometric elements",
            "A visual metaphor for transformation or change",
            "A narrative-driven scene telling a story without words"
        ],
        "Abstract": [
            "Macro photography of natural textures and patterns",
            "Intentional camera movement creating painterly effects",
            "Light and shadow interplay creating geometric forms",
            "Color gradient abstractions with minimal elements",
            "Multiple exposure abstractions blending organic shapes"
        ],
        "Documentary": [
            "An authentic moment capturing daily life",
            "A storytelling image showing human connection",
            "An environmental portrait in a meaningful location",
            "A decisive moment with strong composition",
            "A sequence of images showing process or change"
        ]
    }
    
    if theme in suggestions:
        for suggestion in suggestions[theme]:
            text_widget.insert(tk.END, f"• {suggestion}\n\n")
    else:
        text_widget.insert(tk.END, "No suggestions available for this theme.")

def copy_to_clipboard(text_widget):
    """Copy the content of the text widget to clipboard."""
    content = text_widget.get("1.0", tk.END)
    app.clipboard_clear()
    app.clipboard_append(content)
    messagebox.showinfo("Copy", "Content copied to clipboard!")

def save_suggestions(text_widget):
    """Save the suggestions to a text file."""
    content = text_widget.get("1.0", tk.END)
    if not content.strip():
        messagebox.showwarning("Save", "No content to save!")
        return
    
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Save", "Suggestions saved successfully!")

########################################################################
#         TUTORIALS INTERFACE
########################################################################
def build_tutorials_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    
    # Create a title label
    ttk.Label(frame, text="Tutorials & Help", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    # Create a notebook for different tutorial categories
    tutorial_notebook = ttk.Notebook(frame)
    tutorial_notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    # Getting Started tab
    getting_started = ttk.Frame(tutorial_notebook)
    tutorial_notebook.add(getting_started, text="Getting Started")
    getting_started.columnconfigure(0, weight=1)
    getting_started.rowconfigure(0, weight=1)
    
    gs_text = tk.Text(getting_started, wrap=tk.WORD)
    gs_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    gs_text.insert(tk.END, """
# Welcome to AWDigitalworld Image Analyzer

This application helps you analyze images using AI models, create prompts, and manage your image gallery.

## Getting Started

1. **Image Analysis**: Upload an image to get AI-generated captions, object detection, and more.
2. **Prompt Creator**: Create customized prompts for your images or use templates.
3. **Gallery**: Organize and manage your image collection.
4. **AI Trainer**: Train custom models on your image datasets.

## Tips for Beginners

- Start with the Image Analysis feature to explore what the AI can detect
- Use drag-and-drop to quickly load images
- Save generated prompts for later use
- Explore the Gallery to keep your images organized

Need more help? Check the other tutorial tabs for detailed instructions.
""")
    gs_text.config(state="disabled")
    
    # Image Analysis tab
    analysis_tutorial = ttk.Frame(tutorial_notebook)
    tutorial_notebook.add(analysis_tutorial, text="Image Analysis")
    analysis_tutorial.columnconfigure(0, weight=1)
    analysis_tutorial.rowconfigure(0, weight=1)
    
    analysis_text = tk.Text(analysis_tutorial, wrap=tk.WORD)
    analysis_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    analysis_text.insert(tk.END, """
# Image Analysis Tutorial

The Image Analysis feature uses AI to extract information from your images.

## How to Use

1. Click "Select Image" or drag and drop an image onto the application
2. Wait for the analysis to complete
3. View the results including:
   - AI-generated caption
   - Detected objects with confidence scores
   - Text extracted from the image (OCR)
   - Face detection results
   - EXIF metadata from the camera
   - Color analysis

## Tips for Best Results

- Use clear, well-lit images for more accurate analysis
- For object detection, ensure objects are clearly visible
- For OCR, use images with clear, readable text
- If face detection is important, ensure faces are clearly visible

## Troubleshooting

- If analysis is slow, the AI models are still processing
- If results are not accurate, try different images or angles
""")
    analysis_text.config(state="disabled")
    
    # Other tutorial tabs could be added here
    
    # Button frame
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=2, column=0, pady=10)
    
    ttk.Button(btn_frame, text="Back to Home", command=show_splash_frame).pack(side=tk.LEFT, padx=5)

########################################################################
#           PROMPT CREATOR INTERFACE
########################################################################
def build_prompt_creator_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    ttk.Label(frame, text="Create New Prompt Template", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    dropdown_container = ttk.Frame(frame)
    dropdown_container.grid(row=1, column=0, sticky="nsew")
    global prompt_preview
    option_vars, prompt_preview = build_prompt_creator(dropdown_container)
    btn_creator_frame = ttk.Frame(frame)
    btn_creator_frame.grid(row=2, column=0, pady=10, sticky="ew")
    ttk.Button(btn_creator_frame, text="Save New Prompt", command=save_new_prompt_in_frame).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_creator_frame, text="Back to Home", command=show_splash_frame).pack(side=tk.LEFT, padx=5)

def save_new_prompt_in_frame():
    new_prompt = prompt_preview.get("1.0", tk.END).strip()
    if not new_prompt:
        messagebox.showwarning("Save Prompt", "No prompt generated!")
        return
    opts = getattr(prompt_preview, "generated_options", {})
    insert_prompt(new_prompt, opts)
    messagebox.showinfo("Save Prompt", "New prompt saved successfully!")
    show_splash_frame()

########################################################################
#           PROMPT CREATOR (Embedded) - Dropdowns & Preview
########################################################################
def build_prompt_creator(frame):
    options_data = {
    "Shutter Speed": ["1/1000", "1/500", "1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1", "Bulb"],
    "Scene": ["urban", "nature", "indoor", "studio", "rural", "futuristic", "vintage"],
    "Style": ["realistic", "impressionistic", "cinematic", "documentary", "abstract"],
    "Time of Day": ["morning", "afternoon", "evening", "night"],
    "Photo Style": ["portrait", "landscape", "candid", "macro", "street", "fashion"],
    "Clothing Style": ["casual", "formal", "sport", "punk", "vintage", "bohemian", "gothic", "futuristic"],
    "Eye Color": ["brown", "blue", "green", "hazel", "gray", "amber"],
    "Hair Style": ["straight", "wavy", "curly", "braided", "updo"],
    "Hair Length": ["short", "medium", "long"],
    "Hair Color": ["black", "brown", "blonde", "red", "auburn", "gray"],
    "Age Group": ["18-25", "26-35", "36-45", "46-60", "60+"],
    "Vehicle Type": ["car", "truck", "motorcycle", "bicycle", "scooter", "van", "bus"],
    "Vehicle Style": ["sport", "luxury", "classic", "modern", "off-road", "electric"],
    "Mood": ["happy", "sad", "angry", "calm", "mysterious", "energetic"],
    "Lighting": ["natural", "artificial", "neon", "dramatic", "soft"]
    }
    option_vars = {}
    row_index = 0
    for key, values in options_data.items():
        ttk.Label(frame, text=key+":", font=("Helvetica", 10)).grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=values[0])
        combobox = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
        combobox.grid(row=row_index, column=1, padx=5, pady=2)
        option_vars[key] = var
        row_index += 1

    ttk.Label(frame, text="Figure Measurements:", font=("Helvetica", 10)).grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
    fig_var = tk.StringVar()
    fig_entry = ttk.Entry(frame, textvariable=fig_var)
    fig_entry.grid(row=row_index, column=1, padx=5, pady=2)
    option_vars["Figure Measurements"] = fig_var
    row_index += 1

    ttk.Label(frame, text="Additional Notes:", font=("Helvetica", 10)).grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
    notes_var = tk.StringVar()
    notes_entry = ttk.Entry(frame, textvariable=notes_var)
    notes_entry.grid(row=row_index, column=1, padx=5, pady=2)
    option_vars["Additional Notes"] = notes_var
    row_index += 1

    ttk.Label(frame, text="Generated Prompt:", font=("Helvetica", 10)).grid(row=row_index, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    row_index += 1
    
    prompt_preview = tk.Text(frame, wrap=tk.WORD, height=10)
    prompt_preview.grid(row=row_index, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
    frame.grid_rowconfigure(row_index, weight=1)
    row_index += 1

    def generate_creator_prompt():
        opts = {key: var.get() for key, var in option_vars.items()}
        prompt_lines = [f"{key}: {value}" for key, value in opts.items()]
        final_prompt = "\n".join(prompt_lines)
        prompt_preview.delete("1.0", tk.END)
        prompt_preview.insert(tk.END, final_prompt)
        prompt_preview.generated_options = opts
        prompt_preview.generated_prompt = final_prompt

    def prepopulate_from_docx():
        if not docx_support:
            messagebox.showwarning("DOCX Support", "Python-docx library is not installed. DOCX support is disabled.")
            return
            
        try:
            if not os.path.exists(DOCX_FILE):
                messagebox.showwarning("File Not Found", f"DOCX file not found: {DOCX_FILE}")
                return
                
            document = Document(DOCX_FILE)
            full_text = "\n".join(para.text for para in document.paragraphs)
            prompt_preview.delete("1.0", tk.END)
            prompt_preview.insert(tk.END, full_text)
            prompt_preview.generated_prompt = full_text
            prompt_preview.generated_options = {}
        except Exception as e:
            messagebox.showerror("DOCX Error", f"Error loading DOCX: {e}")
            append_log(f"DOCX error: {e}")

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row_index, column=0, columnspan=2, pady=5)
    ttk.Button(btn_frame, text="Generate Prompt", command=generate_creator_prompt).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Prepopulate from DOCX", command=prepopulate_from_docx).pack(side=tk.LEFT, padx=5)
    row_index += 1

    def save_new_prompt_in_creator():
        if not hasattr(prompt_preview, "generated_prompt") or not prompt_preview.generated_prompt.strip():
            messagebox.showwarning("Save Prompt", "No prompt generated!")
            return
        insert_prompt(prompt_preview.generated_prompt, 
                      getattr(prompt_preview, "generated_options", {}))
        messagebox.showinfo("Save Prompt", "New prompt saved successfully!")
        show_splash_frame()

    ttk.Button(frame, text="Save New Prompt", command=save_new_prompt_in_creator).grid(row=row_index, column=0, columnspan=2, pady=5)

    return option_vars, prompt_preview

########################################################################
#           ADD-ON: PROMPT LIBRARY INTERFACE
########################################################################
def build_prompt_library_interface(frame):
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    ttk.Label(frame, text="Prompt Library", font=("Helvetica", 16)).grid(row=0, column=0, pady=10)
    
    global listbox
    listbox = tk.Listbox(frame)
    listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    scrollbar.grid(row=1, column=1, sticky="ns", pady=5)
    listbox.config(yscrollcommand=scrollbar.set)

    def load_prompts():
        listbox.delete(0, tk.END)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, prompt_text FROM prompts ORDER BY timestamp DESC")
        prompts = c.fetchall()
        conn.close()
        for pid, prompt_text in prompts:
            display = f"{pid}: {prompt_text[:50]}..."
            listbox.insert(tk.END, display)

    load_prompts()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
    ttk.Button(btn_frame, text="Refresh Library", command=load_prompts).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Back to Home", command=show_splash_frame).pack(side=tk.LEFT, padx=5)

########################################################################
#           MAIN APPLICATION SETUP
########################################################################

# Initialize the database first
init_db()

def main():
    global app, style
    
    # Create the main application window
    if drag_drop_support:
        app = TkinterDnD.Tk()
    else:
        app = tk.Tk()
    
    app.title("AWDigitalworld - Offline Image Analyzer")
    app.geometry("1024x726")
    app.rowconfigure(2, weight=0)
    app.rowconfigure(1, weight=1)
    app.columnconfigure(0, weight=1)
    app.bind("<Configure>", lambda event: resize_handler(event))
    
    # Set up style if ttkbootstrap is available
    if ttk_bootstrap:
        style = ttk.Style(theme="superhero")
    
    # Top Navigation Bar
    top_nav = ttk.Frame(app, height=50)
    top_nav.grid(row=0, column=0, sticky="ew")
    top_nav.grid_propagate(False)
    build_top_nav(top_nav)
    
    # Central Content Area
    content_area = ttk.Frame(app)
    content_area.grid(row=1, column=0, sticky="nsew")
    content_area.rowconfigure(0, weight=1)
    content_area.columnconfigure(1, weight=1)
    
    # Left Navigation Bar (Fixed)
    left_nav = ttk.Frame(content_area, width=200)
    left_nav.grid(row=0, column=0, sticky="ns")
    left_nav.grid_propagate(False)
    build_left_nav(left_nav)
    
    # Central Container for Page Switching
    central_container = ttk.Frame(content_area)
    central_container.grid(row=0, column=1, sticky="nsew")
    central_container.rowconfigure(0, weight=1)
    central_container.columnconfigure(0, weight=1)
    
    # Create pages
    global splash_frame, analysis_frame, prompt_frame, training_frame
    global gallery_frame, prompt_library_frame, process_log_frame
    global prompt_suggestions_frame, prompt_rating_frame, tutorials_frame
    
    splash_frame = ttk.Frame(central_container)
    analysis_frame = ttk.Frame(central_container)
    prompt_frame = ttk.Frame(central_container)
    training_frame = ttk.Frame(central_container)
    gallery_frame = ttk.Frame(central_container)
    prompt_suggestions_frame = ttk.Frame(central_container)
    prompt_rating_frame = ttk.Frame(central_container)
    tutorials_frame = ttk.Frame(central_container)
    prompt_library_frame = ttk.Frame(central_container)
    process_log_frame = ttk.Frame(central_container)
    
    pages = (splash_frame, analysis_frame, prompt_frame, training_frame, gallery_frame,
            prompt_suggestions_frame, prompt_rating_frame, tutorials_frame, prompt_library_frame, process_log_frame)
            
    for f in pages:
        f.grid(row=0, column=0, sticky="nsew")
        f.columnconfigure(0, weight=1)
        for i in range(5):
            f.rowconfigure(i, weight=1)
    
    # Build each page
    build_splash_frame(splash_frame)
    build_image_analysis(analysis_frame)
    build_prompt_creator_interface(prompt_frame)
    build_training_interface(training_frame)
    build_gallery_interface(gallery_frame)
    build_prompt_suggestions_interface(prompt_suggestions_frame)
    build_tutorials_interface(tutorials_frame)
    build_prompt_library_interface(prompt_library_frame)
    build_process_log_interface(process_log_frame)
    
    # Bottom Status Bar
    stats_label = ttk.Label(app, text="CPU: --% | RAM: --% | TEMPS: ...", font=("Helvetica", 8))
    stats_label.grid(row=2, column=0, sticky="ew")
    
    def update_system_stats():
        if not system_monitoring:
            stats_label.config(text="System monitoring disabled (psutil not installed)")
            return
            
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            temp_text = "No temperature data"
            
            if hasattr(psutil, "sensors_temperatures"):
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        lines = []
                        for name, entries in temps.items():
                            for e in entries:
                                lines.append(f"{name}({e.label or 'Sensor'}): {e.current}°C")
                        temp_text = " ".join(lines)
                except Exception as e:
                    temp_text = f"Temp error: {e}"
            
            stats_label.config(text=f"CPU: {cpu:.1f}% | RAM: {ram:.1f}% | TEMPS: {temp_text}")
            app.after(1000, update_system_stats)
        except Exception as ex:
            print("Update stats error:", ex)
    
    update_system_stats()
    
    # Load AI models if dependencies are available
    if torch_available and (transformers_available or yolo_available):
        threading.Thread(target=load_models, daemon=True).start()
    
    # Start with the splash screen
    show_splash_frame()
    
    # Start the main event loop
    app.mainloop()

# Build top navigation
def build_top_nav(frame):
    frame.config(height=50)
    frame.grid_propagate(False)
    # Use grid so buttons expand with the window
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="AWDigitalworld Image Analyzer", font=("Helvetica", 16)).grid(row=0, column=0, padx=10, sticky="w")
    nav_buttons = [
        ("Home", show_splash_frame),
        ("Analysis", show_main_frame),
        ("Prompt Creator", show_prompt_frame),
        ("Train Model", show_training_frame),
        ("Gallery", show_gallery_frame),
        ("Log", show_process_log_frame)
    ]
    col = 1
    for (text, cmd) in nav_buttons:
        ttk.Button(frame, text=text, command=cmd).grid(row=0, column=col, padx=5, sticky="ew")
        frame.columnconfigure(col, weight=1)
        col += 1
    ttk.Button(frame, text="Toggle Dark Mode", command=toggle_theme).grid(row=0, column=col, padx=10, sticky="e")
    frame.columnconfigure(col, weight=1)

# Build left navigation
def build_left_nav(frame):
    frame.columnconfigure(0, weight=1)

    ttk.Label(frame, text="Navigation", font=("Helvetica", 12, "bold")).grid(row=0, column=0, pady=(10, 20), sticky="w")

    ttk.Button(frame, text="🏠 Home", command=show_splash_frame).grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    ttk.Button(frame, text="🖼️ Gallery", command=show_gallery_frame).grid(row=2, column=0, sticky="ew", padx=10, pady=5)
    ttk.Button(frame, text="🧠 Prompt Creator", command=show_prompt_frame).grid(row=3, column=0, sticky="ew", padx=10, pady=5)
    ttk.Button(frame, text="🔍 Image Analysis", command=show_main_frame).grid(row=4, column=0, sticky="ew", padx=10, pady=5)
    ttk.Button(frame, text="🧪 AI Trainer", command=show_training_frame).grid(row=5, column=0, sticky="ew", padx=10, pady=5)
    ttk.Button(frame, text="🪵 Log Monitor", command=show_process_log_frame).grid(row=6, column=0, sticky="ew", padx=10, pady=5)
    
    # Show available dependencies in the left nav
    dep_frame = ttk.Frame(frame)
    dep_frame.grid(row=8, column=0, sticky="ew", padx=10, pady=(20, 5))
    ttk.Label(dep_frame, text="Dependencies:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
    
    # Add indicator lights
    row = 1
    for name, status, _ in [
        ("OpenCV", opencv_available, ""),
        ("PyTorch", torch_available, ""),
        ("YOLO", yolo_available, "")
    ]:
        status_color = "green" if status else "red"
        indicator = ttk.Label(dep_frame, text="●", foreground=status_color)
        indicator.grid(row=row, column=0, sticky="w", padx=(0, 5))
        ttk.Label(dep_frame, text=name).grid(row=row, column=1, sticky="w")
        row += 1

if __name__ == '__main__':
    main()