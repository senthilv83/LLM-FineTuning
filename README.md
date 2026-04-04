# Comprehensive 5G Network Slicing & Knowledge Graph LoRA Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-ffcc00.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This repository demonstrates an end-to-end, highly optimized pipeline for **5G Network Slicing** recommendation systems using **Knowledge Graph (KG) integration** and **Large Language Model (LLM) fine-tuning**. 

Specifically, this project utilizes **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)** on the **Llama-3.2-1B-Instruct** model. By converting raw telecommunication tabular data (throughput, latency, reliability) into structured Knowledge Graph triples, we train an AI agent to act as a 5G network slicing expert that maps real-world parameters to 3GPP standards (e.g., TS 23.501) and recommends optimal network configurations.

## 🚀 Key Features

* **Knowledge Graph Construction:** Converts raw 5G dataset parameters into Semantic Subject-Predicate-Object (SVO) triples using `pandas` and visualizes the ontology with `NetworkX`.
* **LLM Prompt Engineering:** Transforms graph data into structured instruction datasets optimized for Llama-3's chat template.
* **LoRA & PEFT Fine-Tuning:** Efficiently fine-tunes the `meta-llama/Llama-3.2-1B-Instruct` model using `trl` (SFTTrainer) and `peft`, drastically reducing GPU VRAM requirements.
* **Apple Silicon (MPS) & CUDA Support:** Dynamically handles precision and quantization. Includes stability workarounds specifically for macOS/MPS hardware.
* **Inference Pipeline:** Merges the base model with the trained LoRA adapter for fast, accurate inference on unseen 5G network conditions.

## 🧠 Technical Architecture

1. **Data Processing:** Parses network parameters (Throughput, Latency, Density, Mobility, Error) and tags them to specific 3GPP standards.
2. **Graph Visualization:** Uses `NetworkX` spring layouts to visualize the relationships between standards, conditions, and network slicing plans.
3. **Model Initialization:** Loads the Llama-3 model in FP16 (or 4-bit NF4 via `bitsandbytes` on CUDA devices).
4. **Supervised Fine-Tuning (SFT):** Employs targeted LoRA configuration on Attention (`q_proj`, `v_proj`, etc.) and MLP layers to optimize domain-specific adaptation.
5. **Evaluation:** Extracts training metrics and plots loss curves. Evaluates substring match accuracy against expected network recommendations.

## 🛠️ Installation & Setup

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install pandas networkx matplotlib torch transformers datasets peft trl bitsandbytes accelerate scikit-learn
```

### Hugging Face Authentication
The Llama-3 model requires authentication. Export your Hugging Face token in your terminal before running the notebooks or scripts:

```bash
export HF_TOKEN="your_hugging_face_token_here"
```

## 📁 Repository Structure

* `Comprehensive_5G_KG_LoRA.ipynb`: The primary, fully-commented Jupyter Notebook detailing the entire pipeline from KG creation to model inference.
* `kg_lora_llama_sft.py`: Standalone Python script for automated batch processing and training.
* `network_slicing_300.csv` / `kg_instruction_data_example.csv`: Sample 5G telemetry and formatted instruction datasets.

## 📈 Search Engine Optimization (SEO) Keywords
*5G Network Slicing, Knowledge Graph, LLM Fine-Tuning, LoRA, Llama-3, Parameter-Efficient Fine-Tuning (PEFT), Telecom AI, PyTorch, Hugging Face Transformers, AI Networking, 3GPP TS 23.501.*

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
