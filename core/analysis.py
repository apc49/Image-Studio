
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

selected_image_path = None
result_text_widget = None

def build_analysis_interface(parent_frame):
    global result_text_widget
    parent_frame.columnconfigure(0, weight=1)

    ttk.Label(parent_frame, text="🧠 Image Analyzer", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

    ttk.Button(parent_frame, text="📂 Load Image", command=select_image).grid(row=1, column=0, pady=5)
    ttk.Button(parent_frame, text="🔍 Analyze", command=analyze_image).grid(row=2, column=0, pady=5)

    result_text_widget = tk.Text(parent_frame, height=10, wrap="word")
    result_text_widget.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

def select_image():
    global selected_image_path
    filetypes = [("Image files", "*.png *.jpg *.jpeg")]
    selected_image_path = filedialog.askopenfilename(filetypes=filetypes)
    if result_text_widget:
        result_text_widget.insert(tk.END, f"Selected: {selected_image_path}\n")

def analyze_image():
    if not selected_image_path:
        result_text_widget.insert(tk.END, "⚠️ No image selected.\n")
        return
    result_text_widget.insert(tk.END, "🔍 Performing analysis (stub)...\n")
    result_text_widget.insert(tk.END, "✅ Analysis complete! (detected: 'cat')\n")
