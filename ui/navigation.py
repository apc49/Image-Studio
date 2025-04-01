
from tkinter import ttk

def build_nav_buttons(container, nav_callback):
    ttk.Button(container, text="🧠 Analyzer", command=lambda: nav_callback("analysis")).pack(pady=5)
    ttk.Button(container, text="🛠 Filters", command=lambda: nav_callback("filters")).pack(pady=5)
    ttk.Button(container, text="🏋️ Trainer", command=lambda: nav_callback("training")).pack(pady=5)
    ttk.Button(container, text="🔍 Validation", command=lambda: nav_callback("validation")).pack(pady=5)
    ttk.Button(container, text="🗂 Dataset Helper", command=lambda: nav_callback("dataset")).pack(pady=5)
