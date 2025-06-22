import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ----- Image Processing Functions -----

def edge_mask(image, line_size=7, blur_value=7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        line_size,
        blur_value
    )
    return edges

def color_quantization(image, k=9):
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    return center[label.flatten()].reshape(image.shape)

def enhance_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# ----- GUI Class -----

class ToonifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Toonify - Cartoon Your Image")
        self.image_rgb = None
        self.cartoon_rgb = None

        self.line_size = tk.IntVar(value=7)
        self.blur_value = tk.IntVar(value=7)
        self.k_clusters = tk.IntVar(value=9)

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        tk.Label(root, text="Edge Line Size").pack()
        tk.Scale(root, from_=3, to=15, orient=tk.HORIZONTAL, variable=self.line_size).pack()

        tk.Label(root, text="Blur Level").pack()
        tk.Scale(root, from_=3, to=15, orient=tk.HORIZONTAL, variable=self.blur_value).pack()

        tk.Label(root, text="Color Clusters (K)").pack()
        tk.Scale(root, from_=3, to=15, orient=tk.HORIZONTAL, variable=self.k_clusters).pack()

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.save_btn = tk.Button(root, text="Save Cartoon Image", command=self.save_image)
        self.save_btn.pack(pady=10)
        self.save_btn.config(state="disabled")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        img_bgr = cv2.imread(file_path)
        self.image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.cartoon_rgb = self.run_toonify(self.image_rgb)
        self.display_image(self.cartoon_rgb)
        self.save_btn.config(state="normal")

    def run_toonify(self, image):
        line = self.line_size.get()
        blur = self.blur_value.get()
        k = self.k_clusters.get()
        edges = edge_mask(image, line_size=line, blur_value=blur)
        quantized = color_quantization(image, k=k)
        smoothed = cv2.bilateralFilter(quantized, d=5, sigmaColor=300, sigmaSpace=300)
        enhanced = enhance_colors(smoothed)
        return cv2.bitwise_and(enhanced, enhanced, mask=edges)

    def display_image(self, image):
        img_pil = Image.fromarray(image).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

    def save_image(self):
        if self.cartoon_rgb is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            cartoon_bgr = cv2.cvtColor(self.cartoon_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, cartoon_bgr)
            messagebox.showinfo("Saved", f"Cartoon image saved to:\n{file_path}")

# ----- Launch App -----

if __name__ == "__main__":
    root = tk.Tk()
    app = ToonifyApp(root)
    root.mainloop()

