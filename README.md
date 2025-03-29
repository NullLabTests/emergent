# Emergent Intelligence Simulator (EIS)

The **Emergent Intelligence Simulator (EIS)** is a Python-based simulation designed to run indefinitely, exploring emergent properties through the interactions of autonomous agents in a dynamic 2D grid environment. Agents evolve, learn, and interact, with a speculative goal of observing complexity that could theoretically lead to Artificial General Intelligence (AGI) or Artificial Superintelligence (ASI).

## Features
- **Environment:** A 100x100 grid with dynamic resources and random environmental events.
- **Agents:** Autonomous entities with neural networks, capable of movement, resource collection, reproduction, and communication.
- **Evolution:** Genetic algorithm with mutation and natural selection based on energy.
- **Learning:** Basic reinforcement learning adjusts agent decision-making.
- **Visualization:** Real-time display using Matplotlib.
- **Emergence:** Designed to foster self-organization and collective behaviors.

## Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Emergent-Intelligence-Simulator.git
   cd Emergent-Intelligence-Simulator
   ```
2. Run the simulation:
   ```bash
   python eis.py
   ```

## How It Works
- Agents navigate a grid, collecting resources to survive and reproduce.
- A simple neural network drives decisions, adapting via rewards and evolution.
- Cooperation emerges through signal-based resource sharing.
- The simulation runs indefinitely, logging population and energy metrics.

## Limitations
- True AGI/ASI is not realistically achievable here; this is an educational tool.
- Performance may degrade with large agent populations due to computational limits.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing
Feel free to fork, experiment, and submit pull requests with improvements!
