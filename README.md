# PERL Planning

This repository contains the code and related experimental materials for the following peer-reviewed article:

> Keke Long, Zhaohui Liang, Haotian Shi, Lei Shi, Sikai Chen, and Xiaopeng Li, "Traffic Oscillation Mitigation with Physics-Enhanced Residual Learning (PERL)-Based Predictive Control," *Communications in Transportation Research*, vol. 4, 2024, Article 100154.  
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772424724000374) | [DOI: 10.1016/j.commtr.2024.100154](https://doi.org/10.1016/j.commtr.2024.100154)

## Overview

This repository implements a **Physics-Enhanced Residual Learning (PERL)**-based predictive control method for vehicle trajectory prediction and control, particularly designed for mixed traffic environments involving both connected and automated vehicles (CAVs) and human-driven vehicles (HDVs).

The PERL model combines physical information, specifically traffic-wave properties, with data-driven features extracted through deep learning. By predicting the behavior of preceding vehicles, especially speed fluctuations, the system helps CAVs respond in advance, reducing traffic oscillations and improving safety and comfort.

The system consists of two tasks:

1. **Prediction Model**: Predicts the future behavior of preceding vehicles.
2. **CAV Controller**: Uses Model Predictive Control (MPC) to improve the safety and efficiency of the vehicle platoon.

The method reported in the corresponding article was tested through vehicle-in-the-loop (ViL) experiments and compared with real driving data and three benchmark models.

## Project Structure

- **Prediction Model**: Implementation of PERL for predicting vehicle behavior.
- **Data Preparation**: Scripts for processing vehicle trajectory data.
  - `1 prepare chain trj.py`: Prepares trajectory data for the prediction model.
  - `2 prepare_acceleration prediction.py`: Prepares data for acceleration prediction.
- **Planning and Control**: MPC-based control strategies.
  - `3 planning_MPC.py`: Implements MPC for trajectory planning.
  - `4 planning_HV.py`: Implements a baseline planning model for human-driven vehicles.
- **Results**:
  - `experiment results/`: Experimental outputs.
  - `platooning results/`: Platooning simulation results.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/Keke-Long/PERL_planning.git
   cd PERL_planning
   ```

2. Run the relevant data-preparation scripts before executing the planning algorithms. For example:

   ```bash
   python "1 prepare chain trj.py"
   ```

## Citation

If you use this repository, please cite:

```bibtex
@article{long2024traffic,
  title={Traffic oscillation mitigation with physics-enhanced residual learning (PERL)-based predictive control},
  author={Long, Keke and Liang, Zhaohui and Shi, Haotian and Shi, Lei and Chen, Sikai and Li, Xiaopeng},
  journal={Communications in Transportation Research},
  volume={4},
  pages={100154},
  year={2024},
  doi={10.1016/j.commtr.2024.100154}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
