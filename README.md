# Dynamic Inference Adaptation

This repository provides the full codebase for **dynamic inference adaptation**, with the complete **NeuroTTA pipeline**. It investigates how neural networks can adapt their behavior **at inference time** under changing data distributions. The code includes training on CIFAR-10 (baseline models), Test-Time Adaptation (TTA) with dynamic neuron routing, calibration with T* for robust predictions, ablation studies to evaluate each component, and a single-cell Colab pipeline for fast reproduction.

## Getting Started
Clone the repository and install the required dependencies:
git clone https://github.com/<your-username>/dynamic-inference-adaptation.git  
cd dynamic-inference-adaptation  
pip install -r requirements.txt  

Run the full pipeline:
python main.py --config configs/neurotta.yaml  

Repository structure:
configs/ → YAML configs for experiments  
models/ → Model architectures  
adaptation/ → Test-time adaptation modules  
routing/ → Router and calibration  
experiments/ → Ablation and evaluation scripts  
main.py → Entry point  

## Citation
If you use this repository for academic reference, please cite:
@misc{patane2025dynamic,  
  title  = {Dynamic Inference Adaptation},  
  author = {Rosario Patanè},  
  year   = {2025},  
  url    = {https://github.com/<your-username>/dynamic-inference-adaptation}  
}

## License
© 2025 Rosario Patanè. All rights reserved.  
This code is provided for research and personal use only. **Redistribution, modification, or commercial use is not allowed without explicit permission from the author.**

