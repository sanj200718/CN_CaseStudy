# Machine Learning-Based Intelligent Routing Framework for Dynamic Networks
**Group 4 Case Study** | **Team Members:** Karthik R Nair, Iraban Bhattacharya, Cherukuri Sanjana

This repository contains the Machine Learning / Deep Reinforcement Learning (DQN) routing agent developed for dynamic network optimization and congestion avoidance.

---

## Project File Structure
* **`env.py`**: Custom Gymnasium network environment simulating graph topologies with dynamic traffic delay shifts.
* **`model.py`**: PyTorch Deep Q-Network (DQN) architecture.
* **`train.py`**: Training pipeline utilizing experience replay, epsilon decay, and valid action masking.
* **`inference.py`**: Integration wrapper for the SDN/Mininet controller to query real-time next-hop routing decisions.
* **`demo_compare.py`**: Evaluation script comparing the trained DQN agent against standard Dijkstra routing.
* **`routing_dqn.pt`**: Saved pre-trained model weights.
* **`learning_curve.png`**: Convergence graph illustrating agent training performance.

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch gymnasium networkx matplotlib numpy