# rl_project
# RL-Based Chip Placement Using Graph Neural Networks
## Overview

This project explores reinforcement learning for automated VLSI chip placement optimization. The circuit is represented as a graph, where circuit components are modeled as nodes and their connectivity is represented through edges. A Graph Neural Network (GNN) is used to learn circuit representations, while a Proximal Policy Optimization (PPO) agent learns placement decisions through interaction with the placement environment.

The objective is to optimize cell placement while minimizing physical design metrics such as Half-Perimeter Wirelength (HPWL) and routing congestion.
## Overview

This project explores reinforcement learning for automated VLSI chip placement optimization. The circuit is represented as a graph, where circuit components are modeled as nodes and their connectivity is represented through edges. A Graph Neural Network (GNN) is used to learn circuit representations, while a Proximal Policy Optimization (PPO) agent learns placement decisions through interaction with the placement environment.

The objective is to optimize cell placement while minimizing physical design metrics such as Half-Perimeter Wirelength (HPWL) and routing congestion.
## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Graph Neural Networks (GNN)
- Proximal Policy Optimization (PPO)
- Reinforcement Learning
- OpenROAD
- Verilog
- DEF / LEF
## Methodology

### 1. Graph Construction

The circuit netlist is converted into a graph representation for processing by the Graph Neural Network. Circuit cells are represented as nodes, while the connectivity between cells through circuit nets is represented as edges. Node and edge information is extracted from the input design data and used to construct the graph.
### 2. Graph Neural Network

The constructed circuit graph is processed using a Graph Neural Network (GNN) to learn representations of the circuit components and their connectivity. The GNN aggregates information from neighboring nodes and generates embeddings that capture both local cell characteristics and connectivity information. These learned graph embeddings are provided to the reinforcement learning agent as part of the state representation.


### 3. Reinforcement Learning Environment

The placement problem is formulated as a reinforcement learning environment. The environment maintains the current placement state of the circuit and allows the agent to perform placement actions. After each action, the environment updates the placement and evaluates the resulting layout using physical design metrics.

### 4. PPO Agent

A Proximal Policy Optimization (PPO) agent is used to learn an effective placement policy. The agent consists of an Actor and a Critic network. The Actor determines the placement action based on the current state representation, while the Critic estimates the value of the current state. PPO updates the policy using the rewards obtained from the placement environment while maintaining stable policy updates.

### 5. Placement Actions

The agent selects placement actions based on the current circuit state and learned graph representation. The selected action modifies the placement of circuit cells within the available placement region. The updated placement is then evaluated by the environment to determine its effect on the overall layout quality.

### 6. Reward Function

The reward function evaluates the quality of the placement produced by the agent. Physical design metrics such as Half-Perimeter Wirelength (HPWL) and routing congestion are used to measure placement quality. The reward guides the PPO agent toward placement decisions that reduce wirelength and congestion while improving the overall quality of the layout.

### 7. Training

During training, the PPO agent interacts repeatedly with the placement environment. For each step, the agent observes the current state, selects an action, receives a reward, and observes the resulting state. The collected experience is used to update the Actor and Critic networks. This process is repeated over multiple episodes to progressively improve the placement policy.

## Dataset

The project uses circuit design data containing information about cells, nets, connectivity, and initial placement. The input designs are processed to construct graph representations suitable for GNN-based learning and reinforcement learning.

The repository can include sample datasets for demonstration, while larger design files can be prepared separately due to their size.

## Installation

Clone the repository and install the required Python dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Usage

The general workflow is:

```text
Prepare Design Data
        ↓
Construct Circuit Graph
        ↓
Generate Graph Features
        ↓
Initialize RL Environment
        ↓
Train PPO Agent
        ↓
Evaluate Placement
        ↓
Analyze HPWL and Congestion
```

Detailed training and evaluation commands will be provided as the implementation is organized.

## Project Structure

```text
rl-chip-placement/
│
├── data/              # Circuit and processed graph data
├── src/               # Main project source code
│   ├── environment/   # RL environment
│   ├── models/        # GNN and RL models
│   ├── agent/         # PPO agent
│   └── utils/         # Utility functions
│
├── scripts/           # Data processing and evaluation scripts
├── configs/           # Training and experiment configurations
├── notebooks/         # Experiments and analysis
├── outputs/           # Results and visualizations
├── tests/             # Testing
├── requirements.txt   # Python dependencies
└── README.md
```

## Results

The performance of the trained placement agent can be evaluated using physical design metrics such as:

* Half-Perimeter Wirelength (HPWL)
* Routing Congestion
* Placement Quality
* Training Reward

Results and placement visualizations will be added to this section.

## Future Work

* Improve congestion-aware placement optimization
* Incorporate additional physical design metrics into the reward function
* Explore larger and more complex circuit designs
* Improve GNN architecture and graph representation
* Investigate thermal-aware placement optimization
* Compare the proposed approach with conventional placement techniques

## References

This project is inspired by research in reinforcement learning, graph neural networks, and automated VLSI physical design.

Relevant research papers, repositories, and references will be listed here.

## Author

**Asish Vibhu Velampalli**

B.Tech – Electronics and Communication Engineering
