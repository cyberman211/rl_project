# rl_project
# RL-Based Chip Placement Using Graph Neural Networks
## Overview

This project explores reinforcement learning for automated VLSI chip placement optimization. The circuit is represented as a graph, where circuit components are modeled as nodes and their connectivity is represented through edges. A Graph Neural Network (GNN) is used to learn circuit representations, while a Proximal Policy Optimization (PPO) agent learns placement decisions through interaction with the placement environment.

The objective is to optimize cell placement while minimizing physical design metrics such as Half-Perimeter Wirelength (HPWL) and routing congestion.
