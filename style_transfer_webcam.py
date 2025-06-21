#!/usr/local/bin/python3.10
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading

# Load the pre-trained model from TensorFlow Hub
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Style transfer function with softer effect
def apply_style_transfer(hub_module, content_image, style_image, alpha=0.5):
    content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)
    style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
    content_tensor = tf.expand_dims(content_tensor, axis=0)
    style_tensor = tf.expand_dims(style_tensor, axis=0)
    outputs = hub_module(content_tensor, style_tensor)
    stylized_image = outputs[0].numpy()[0]
    # Blend the original content image with the stylized image
    blended_image = alpha * stylized_image + (1 - alpha) * content_image
    return blended_image

# Preprocess image
def preprocess_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size)
    return resized_image / 255.0  # Normalize to [0, 1]

# Main application class
class StyleTransferApp:
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
                self.style_image = preprocess_image(self.style_image)
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
            content_image = preprocess_image(frame)

            # Apply style transfer with softer effect
            stylized_frame = apply_style_transfer(hub_module, content_image, self.style_image)
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

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
