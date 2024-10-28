# PERL Planning

## Overview

This repository implements a **Physics-Enhanced Residual Learning (PERL)**-based predictive control method for vehicle trajectory prediction and control, particularly designed for mixed traffic environments involving both connected and automated vehicles (CAVs) and human-driven vehicles (HDVs). 

The PERL model combines physical information, specifically traffic wave properties, with data-driven features extracted through deep learning techniques. By predicting preceding vehicle behavior, especially speed fluctuations, the system helps CAVs respond in advance, reducing traffic oscillations and improving both safety and comfort.

The system consists of two tasks:
1. **Prediction Model**: Predicts the future behavior of the preceding vehicles.
2. **CAV Controller**: Uses Model Predictive Control (MPC) to improve the safety and efficiency of the entire vehicle platoon.

This project has been tested through vehicle-in-the-loop (ViL) experiments and compared against real driving data and three benchmark models.

## Project Structure

- **Prediction Model**: Contains the implementation of PERL for predicting vehicle behavior.
- **Data Preparation**: Scripts for processing vehicle trajectory data.
  - `1_prepare_chain_trj.py`: Prepares trajectory data for the prediction model.
  - `2_prepare_acceleration_prediction.py`: Prepares data for predicting vehicle acceleration.
- **Planning and Control**: Implements the MPC-based control strategies.
  - `3_planning_MPC.py`: Implements MPC for trajectory planning.
  - `4_planning_HV.py`: Implements a baseline planning model for human-driven vehicles (HV).
- **Results**: Stores experiment and plotting results.
  - `experiment_results/`: Contains outputs from the experiments.
  - `platooning_results/`: Results from platooning simulations.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Keke-Long/PERL_planning.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the data preparation scripts before executing planning algorithms. Example:
   ```bash
   python 1_prepare_chain_trj.py
