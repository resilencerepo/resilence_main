# Moving Soundscapes – Change Analysis

This repository contains the **central control algorithm** for the *Moving Soundscapes* installation.

The patch and demo explore how an incoming stream of data (e.g. motion capture, audio, video) is classified as **intrusive** or **non-intrusive**, based on dynamic qualities like smoothness, jerkiness, and strangeness.

🎥 **Watch the demo**:  
[![Moving Soundscapes Demo](https://img.youtube.com/vi/Kp8f3Qj1T3s/hqdefault.jpg)](https://youtu.be/Kp8f3Qj1T3s)

---

## 🧠 Algorithm Description

- **Input**: A mono channel of integers from `0` to `127`, sampled at `10 ms` intervals.
- **Output**: A label ranging on a continuum from *non-intrusive* to *intrusive*.

---

## 🧩 Patch Structure

### 1. **change_analysis_2028.03.10.maxpat**

- Demo interface includes:
  - **INPUT slider** (simulated input)
  - **OUTPUT slider** (result of analysis)

### 2. **Analysis Stages**

#### 🔹 Level 1: Frame-Based Analysis
- **QM**: Quantity of change
- **accumul**: Change accumulation
- **dir**: Direction detection (up/down/static)
- **smoothness**: Difference between smoothed and original signal
- **scattosità**: Jerkiness

➡️ Outputs qualities like: `smooth`, `medium`, `not smooth`

#### 🔹 Level 2: Abstract Parameters
- Transforms Level 1 results into:
  - `strangeness`
  - `salience`

#### 🔹 Level 3: Integration & Output
- Parameters are averaged in the `statistics` patch
- Final output passes through a **leaky integrator** to decay over time

---

## 📁 Files Included

- `change_analysis_2028.03.10.maxpat` — Main MaxMSP patch
- `change_analysis_demo_2025.03.10.mp4` — Demo video of the patch behavior
- `doc_Moving-Soundscapes_change_analysis_2025.05.17.pdf` — Detailed instructions for setting up and using the Max/MSP patch

---

## 🔗 Resources

📺 [YouTube Demo](https://youtu.be/Kp8f3Qj1T3s)

---

© Andrea Cera 2025
