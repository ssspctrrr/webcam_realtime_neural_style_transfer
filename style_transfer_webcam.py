import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading
from process_image import ProcessImage

# Main application class
class StyleTransferApp(ProcessImage):
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Style Transfer")
        self.style_image = None
        self.running = False

        # UI Elements
        self.label = tk.Label(self.root, text="Yunus\'s Style Transfer AI", font=("comic sans ms", 21))
        self.label.pack(pady=10)
        
        self.label = tk.Label(self.root, text="Load a style image to start", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.load_button = tk.Button(self.root, text="Load Style Image", command=self.load_style_image, width=20)
        self.load_button.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam, width=20, state=tk.DISABLED)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, width=20, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)

    def load_style_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            style_image = cv2.imread(file_path)
            if style_image is not None:
                self.style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
                self.style_image = self.preprocess_image(self.style_image)
                messagebox.showinfo("Success", "Style image loaded successfully!")
                self.start_button.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Failed to load the style image.")

    def start_webcam(self):
        if self.style_image is None:
            messagebox.showerror("Error", "Please load a style image first.")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Start webcam processing in a separate thread
        threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_webcam(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def process_webcam(self):
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Webcam not accessible.")
                break

            # Preprocess webcam frame
            content_image = self.preprocess_image(frame)

            # Apply style transfer with softer effect
            stylized_frame = self.apply_style_transfer(content_image, self.style_image)
            stylized_frame = (stylized_frame * 255).astype(np.uint8)  # Convert back to [0, 255]

            # Convert to PIL Image for Tkinter
            stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(stylized_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.update_idletasks()
            self.root.update()

        cap.release()