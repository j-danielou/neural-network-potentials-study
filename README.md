# TER Project: Learning Interatomic Potentials and their Gradients

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

Welcome to the repository of my TER (*Travail d'Étude et de Recherche* / Research Study Project). This project explores the use of Deep Learning to simultaneously predict the energy of a system and its associated forces.

In molecular dynamics, force is defined as the negative gradient of the energy with respect to atomic coordinates:
$$F = -\nabla E$$

The goal of this project is to evaluate different training strategies (Classic, Auto-differentiation, Multitask) to optimize the learning of energy and its derivatives. We start with a simple mathematical problem before scaling up to complex molecules.

---

## 🔬 Scientific Approach

The project is divided into two progressive phases:

| Phase | Study Environment | Main Objective |
| :--- | :--- | :--- |
| **1. Theory** | Himmelblau Function (Toy Function) | Rigorously understand and validate different gradient learning strategies, study the impact of hyperparameters ($\lambda$), and validate the results via statistical tests (Mann-Whitney). |
| **2. Application** | MD17 Dataset (Molecules) | Apply the validated concepts to real molecules (Paracetamol, Aspirin, ect...) using the **TorchANI** architecture. Study the impact of loss function weights ($\alpha$ and $\beta$). |

---

## 📂 Repository Architecture

The code is organized chronologically to follow the scientific reasoning:

```text
├── notebooks/                   
│   ├── Phase_1_Himmelblau/
│   │   ├── 01_training_strategies_himmelblau.ipynb
│   │   ├── 02_lambda_finetuning_himmelblau.ipynb
│   │   ├── 03_loss_by_epoch_classic.ipynb         
│   │   ├── 04_loss_by_epoch_multitask.ipynb       
│   │   ├── 05_loss_by_epoch_gradient.ipynb        
│   │   └── 06_statistical_analysis_himmelblau.ipynb
│   │
│   └── Phase_2_Molecules/
│       ├── 07_training_ANI_molecule.ipynb
│       ├── 08_loss_by_epoch_ANI_gradient.ipynb    
│       └── 09_optimisation_alpha_beta.ipynb       
│
├── dashboards/                  
│   ├── dashboard_himmelblau.py  # Interactive visualization for Phase 1
│   └── dashboard_molecule.py    # Interactive visualization for Phase 2
│          
├── csv/
│   ├── metrics_md17_7reps_aspirin.csv
│   ├── metrics_md17_7reps_benzene.csv
│   └── metrics_md17_7reps_paracetamol.csv
│                 
└── README.md
