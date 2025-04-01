
import tkinter as tk
from tkinter import ttk, filedialog
import os
import csv

def build_dataset_interface(frame):
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="🗂 Dataset Helper", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

    ttk.Button(frame, text="📁 Browse Dataset Folder", command=scan_folder).grid(row=1, column=0, pady=5)

    global log
    log = tk.Text(frame, height=10, wrap="word")
    log.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

def scan_folder():
    folder = filedialog.askdirectory()
    if not folder:
        return

    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    metadata = []

    for img in images:
        label = os.path.splitext(img)[0].split("_")[0]
        metadata.append((img, label))
        log.insert(tk.END, f"🖼 {img} => Tag: {label}\n")

    csv_path = os.path.join(folder, "dataset_metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(metadata)

    log.insert(tk.END, f"✅ Metadata saved to {csv_path}\n")
