# Edge AI Wildlife Detection & Monitoring System

A lightweight, edge-first wildlife monitoring solution using **YOLOv8 Nano + CBAM Attention**, **OpenCV**, and **SQLite**, designed to run on resource-constrained platforms like NVIDIA Jetson Nano or Raspberry Pi 5.

## Features
- Real-time wildlife and human intrusion detection
- CBAM Attention Module integration support
- Threat Score Engine (Low / Medium / High / Critical)
- Environmental Robustness (CLAHE auto-enhancement for low light)
- Offline-first SQLite event logging & event sync queue
- Streamlit Dark Forest UI dashboard with metrics & downloadable reports