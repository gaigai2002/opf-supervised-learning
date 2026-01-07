# Learning Neural OPF Policies with Supervised and Reinforcement Learning

This repository demonstrates how **AI learning algorithms** can be used to
learn decision-making policies for power system operation.

By combining **supervised learning** and **reinforcement learning (RL)**,
we train neural networks that approximate optimal power flow (OPF) solutions
and enable fast, learning-based control.

---

## 🚀 Project Overview

The project follows a **two-stage learning pipeline**:

### **Stage 1 — Supervised Learning (Imitation Learning)**

- A neural network is trained to **imitate OPF solutions**
- OPF acts as an *expert demonstrator*
- The model learns a mapping from system load conditions to optimal voltage states

**Input**
- Active and reactive load demands at all buses

**Output**
- Bus voltage magnitudes  
- Bus voltage phase angles

This stage provides a strong and stable initialization for control tasks.

---

### **Stage 2 — Reinforcement Learning (Policy Optimization)**

- The pretrained model can be embedded into a **reinforcement learning environment**
- An agent interacts with a simulated power grid
- Rewards are defined based on:
  - Operational cost
  - Physical constraint violations
- The learned policy enables fast decision-making without solving OPF online

---

## 🧠 Learning Paradigm

This project integrates multiple AI paradigms:

- **Supervised Learning**  
  Learning from optimal OPF demonstrations

- **Reinforcement Learning**  
  Learning control policies through interaction and reward feedback

- **Neural Networks**  
  Fully connected architectures with constrained outputs

The hybrid approach combines **training stability** and **control flexibility**.

---

## ✨ Key Features

- Neural surrogate for OPF solutions
- Physically constrained network outputs
- Modular simulation environment
- RL-ready design for control research

---

## 📌 Applications

- Fast OPF approximation
- Learning-based power system control
- AI-assisted optimization
- Reinforcement learning research in energy systems

---

## ▶️ Quick Start

```bash
pip install -r requirements.txt
python env.py 
python pretrain_fn.py

---

## 📁 Repository Structure

```text
.
├── env.py           # Power grid simulation environment
├── agent1_va.py     # Neural network models
├── pretrain_fn.py   # Supervised learning pipeline
├── saved_data/      # Example training dataset
├── saved_model/     # Trained models
├── training_plots/  # Training curves
├── requirements.txt # Environment dependencies
└── README.md
