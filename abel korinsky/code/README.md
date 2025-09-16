# Vehicular Soundscapes Software

## Overview
This repository contains a Python-based software system built around **computer vision, audio targeting, and AI-driven analysis**.  
It integrates detection, tracking, analysis, and generative storytelling into an interactive, modular architecture.

The software consists of **four interlinked modules**:

---

## 🔹 1. Detection & Tracking Module
- Uses **YOLOv8** for state-of-the-art object detection (optimized for pedestrians).  
- Employs **ByteTrack** for multi-object tracking, ensuring persistent and reliable IDs even in crowded scenes.  
- Displays live tracking overlays.  
- Randomly selects targets on **Monitor 1** for further processing.  

---

## 🔹 2. Audio Targeting Module
- Synchronizes tracking data with **ultrasonic beam steering**.  
- Converts camera-detected coordinates into **real-world angular values**.  
- Steers sound beams to target individuals with precise audio delivery.  

---

## 🔹 3. Person Analysis Module
- Captures **snapshots** of selected individuals.  
- Performs **local AI-based analysis** of:
  - Race  
  - Age  
  - Gender  
  - Clothing  
  - Facial expression  
- Runs **fully on-device** to ensure **GDPR compliance** (no cloud processing).  

---

## 🔹 4. Story Generation Module
- Uses a **locally hosted language model** to generate **speculative micro-narratives** about tracked individuals.  
- Narratives are output as:
  - **Synthesized voice**  
  - **Visual display on Monitor 2**  

---

## 🛠️ Tools & Methodologies
- **YOLOv8** – Object detection  
- **ByteTrack** – Multi-object tracking  
- **Ultrasonic beam steering** – Audio targeting  
- **Local AI models** – Feature analysis (privacy-preserving)  
- **Local LLM** – Story generation  

---

## ⚙️ System Architecture
The modules interact in a pipeline:  

1. **Detection & Tracking** →  
2. **Audio Targeting + Person Analysis** →  
3. **Story Generation + Output**

This modular architecture ensures flexibility, scalability, and strict **data privacy** principles.  

---

## 🚀 Getting Started
### Prerequisites
- Python 3.9+  
- Dependencies (see `requirements.txt`)  
- YOLOv8 weights  
- Local AI models (instructions in `/models` folder)  

### Installation
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
