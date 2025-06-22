# 🖼️ Toonify - Cartoon Your Images

Toonify is a Python application that transforms regular photos into cartoon-style images using OpenCV and a simple GUI made with Tkinter.

## ✨ Features

- 🖼️ Upload any JPG or PNG image
- 🎛️ Adjust cartoon style with sliders:
  - Edge Line Thickness
  - Blur Strength
  - Color Simplification (K Clusters)
- 💾 Save the cartoonified image
- 📦 Packageable as `.exe` for Windows

## 📸 How It Works

Toonify uses a combination of:
- **Edge detection** using adaptive thresholding
- **Color quantization** using K-Means
- **Bilateral filtering** for smoothing
- **Color enhancement** in HSV

## 🚀 Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/Anujshah325/toonify.git
   cd toonify

