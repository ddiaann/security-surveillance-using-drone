# security-surveillance-using-drone
Source code for multi-camera drone surveillance and zero-shot person re-identification.

Transformer-Based Zero-Shot Person Re-Identification

1. Project Overview
This project implements an integrate security surveillance system that leverages both a static ground camera and an autonomous drone. The system is designed to identify and track individuals across different camera views (Person Re-ID) without prior training on specific individuals, utilizing a Transformer-based Zero-shot model.

2. Key Features
   - Multi-Perspective Surveillance: Combines fixed angle security footage with dynamic aerial views from a drone.
   - Zero-Shot Re-ID: Employs a Trasnformer architecture to identify persons across non-overlapping camera without needing identity-specific fine-tuning.
   - Scalaable Architecture: Designed with distributed computing principles to handle high throughput video streams.
   - Real-Time Processing: Optimized for low-latency inference to ensure timely security alerts.
  
3. Technical Stack
   - Vision Models: Vision Transformer (ViT) for feature extraction.
   - Hardware: DJI Tello Drone, Static IP Camera.
   - Language & Frameworks: Python, PyTorch/TensorFLow, OpenCV.
   - Parallel Computing: MPI (Message Passing Interface) for distributed workload handling (if applicable to backend)
  
4. System Architecture
The system operates in three main phases:
   1. Detection: Static camera detects a subject of interest.
   2. Handover: Coordination logic signals to drone to intercept or follow.
   3. Re-IDentification: The Transformer model geenrates embeddings to match the subject's identity between the drone's aerial feed and the ground camera's static feed.
