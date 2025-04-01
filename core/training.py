
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import os

log_output = None
dataset_path = None

def build_training_interface(frame):
    global log_output
    frame.columnconfigure(0, weight=1)

    ttk.Label(frame, text="🏋️ AI Model Trainer", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

    ttk.Button(frame, text="📂 Select Dataset Folder", command=select_dataset).grid(row=1, column=0, pady=5)

    ttk.Button(frame, text="🚀 Start Training", command=start_training).grid(row=2, column=0, pady=5)

    log_output = tk.Text(frame, height=15, wrap="word")
    log_output.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

def select_dataset():
    global dataset_path
    dataset_path = filedialog.askdirectory()
    if log_output:
        log_output.insert(tk.END, f"📁 Dataset selected: {dataset_path}\n")

def start_training():
    if not dataset_path:
        log_output.insert(tk.END, "⚠️ Please select a dataset folder first.\n")
        return

    log_output.insert(tk.END, "📦 Starting training...\n")
    thread = threading.Thread(target=simulate_training)
    thread.start()

def simulate_training():
    for epoch in range(1, 6):
        log_output.insert(tk.END, f"Epoch {epoch}/5 - Loss: {0.9 - epoch * 0.1:.2f}\n")
        log_output.see(tk.END)
        time.sleep(1)
    log_output.insert(tk.END, "✅ Training complete.\n")
