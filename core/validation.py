
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os

def build_validation_interface(frame):
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="🔍 Validation Preview", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

    ttk.Button(frame, text="📂 Select Image", command=lambda: select_and_preview(frame)).grid(row=1, column=0, pady=5)

def select_and_preview(frame):
    path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not path:
        return

    img = Image.open(path)
    img.thumbnail((512, 512))
    tk_img = ImageTk.PhotoImage(img)

    img_label = ttk.Label(frame, image=tk_img)
    img_label.image = tk_img
    img_label.grid(row=2, column=0, pady=10)

    log = tk.Text(frame, height=6, wrap="word")
    log.insert(tk.END, f"📄 File: {os.path.basename(path)}\n")
    log.insert(tk.END, f"🧠 Predicted: object (demo)\n")
    log.grid(row=3, column=0, padx=10, sticky="ew")
