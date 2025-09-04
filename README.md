# 🐍 AI-Powered Snake Game using Deep Reinforcement Learning

This project features an AI agent that masters the classic Snake game from scratch. Using **Deep Q-Learning**, the agent learns optimal strategies through trial and error, with no hard-coded rules. It evolves from random movements to complex, intelligent behaviors, demonstrating a practical application of modern reinforcement learning techniques.


## 🎥 Watch the Agent in Action
(./assets/Screenshot 2025-09-04 132621.png)


## 🛠️ Core Concepts & Technology

This project is built on the principles of **Reinforcement Learning (RL)**. The agent learns by interacting with its environment in a loop: it observes a state, performs an action, and receives a reward.

### 1. State Representation
The agent doesn't "see" the game screen. Instead, it perceives the environment through an **11-element state vector**:

- **Danger Analysis (3 booleans):** Is there an immediate collision danger if the agent moves straight, right, or left?
- **Direction of Motion (4 booleans):** One-hot encoded vector representing the current direction (Up, Down, Left, Right).
- **Food Location (4 booleans):** Indicates if the food is located to the agent’s left, right, up, or down.

### 2. The Model: A Neural Network Brain
A neural network is used as a **function approximator** for Q-values.

- **Architecture:** Input layer (11 neurons) → Hidden layer (256 neurons) → Output layer (3 neurons: straight, right, left).  
- **Framework:** [PyTorch](https://pytorch.org).

### 3. The Learning Algorithm: Deep Q-Learning
The agent uses **Deep Q-Learning** with:

- **Experience Replay:** Stores thousands of past experiences in memory and samples random mini-batches for training, reducing correlations.
- **Epsilon-Greedy Strategy:** Balances exploration (random moves) and exploitation (choosing the best-known action).

---

## 🚀 Getting Started

### Prerequisites
You will need Python 3.7+ and the following libraries:

```bash
pip install pygame torch numpy matplotlib
```
### Installation & Execution
- Clone the repository:
```bash
git clone https://github.com/your-username/snake-ai-project.git
cd snake-ai-project
```
- Run the training script:
```bash
python agent.py
```
- A Pygame window will open where you can watch the agent learn in real-time.
- A plot of score and mean score will also update live.
- The trained model (model.pth) will be saved in the ./model directory whenever a new high score is achieved.

## 📁 Project Structure
```
├── agent.py          # The main training loop and agent logic
├── game.py           # The Snake game environment (Pygame)
├── model.py          # The PyTorch neural network model and Q-trainer
├── game_plot.py      # Real-time plotting of training progress
├── model/            # Directory where the trained model is saved
└── README.md         # This file
```

## ⚙️ Hyperparameter Tuning
Key hyperparameters (set in agent.py):
- MAX_MEMORY: Maximum size of replay memory
- BATCH_SIZE: Mini-batch size for training
- LR: Learning rate (Adam optimizer)
- GAMMA: Discount factor for future rewards

## 📈 Results & Performance
The agent’s progress is visualized via training curves:
- Early phase: Random, low-scoring games
- Breakthrough: Learns to avoid immediate death
- Mastery: Improves steadily, learning to hunt food and avoid collisions

## 💡 Future Improvements
Possible extensions:
- CNN-based Agent: Use raw pixels as input instead of state vector
- Advanced RL Algorithms: Double DQN, Dueling DQN, etc.
- Enhanced State Representation: Include snake’s body, wall distances, etc.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check the issues page
 or open a pull request.

## 🙏 Acknowledgments
- Inspired by DeepMind’s foundational work in reinforcement learning
- Thanks to the creators of Pygame, PyTorch, NumPy, and Matplotlib
- Special appreciation to the open-source AI community
