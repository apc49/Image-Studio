
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image
import os

def build_filters_interface(frame):
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="🛠 Image Filters & Resizer", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

    ttk.Button(frame, text="📂 Choose Image", command=resize_prompt).grid(row=1, column=0, pady=5)

def resize_prompt():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path)
    for size in [(512, 512), (1024, 1024)]:
        resized = img.resize(size, Image.LANCZOS)
        base, ext = os.path.splitext(file_path)
        save_path = f"{base}_{size[0]}x{size[1]}{ext}"
        resized.save(save_path)
        print(f"✅ Saved: {save_path}")
