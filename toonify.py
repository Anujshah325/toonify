import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Failed to load image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def edge_mask(image, line_size=7, blur_value=7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   line_size,
                                   blur_value)
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

def toonify_image(filepath):
    img = read_image(filepath)
    edges = edge_mask(img)
    quantized = color_quantization(img)
    smoothed = cv2.bilateralFilter(quantized, d=5, sigmaColor=300, sigmaSpace=300)
    enhanced = enhance_colors(smoothed)
    cartoon = cv2.bitwise_and(enhanced, enhanced, mask=edges)

    # Display results
    titles = ["Original", "Edges", "Enhanced", "Cartoon"]
    images = [img, edges, enhanced, cartoon]
    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 1 else None)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return cartoon

if __name__ == "__main__":
    image_path = "images/sample.jpg"
    cartoon = toonify_image(image_path)
    cv2.imwrite("toonified_output.jpg", cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
