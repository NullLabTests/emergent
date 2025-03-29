import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import randint, choice, uniform

# Simple Neural Network for Agent Decision-Making
class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.uniform(-1, 1, (3, 5))  # 3 inputs -> 5 hidden
        self.weights2 = np.random.uniform(-1, 1, (5, 3))  # 5 hidden -> 3 outputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden = self.sigmoid(np.dot(inputs, self.weights1))
        output = self.sigmoid(np.dot(hidden, self.weights2))
        return output

    def mutate(self):
        if uniform(0, 1) < 0.05:
            self.weights1 += np.random.uniform(-0.1, 0.1, self.weights1.shape)
            self.weights2 += np.random.uniform(-0.1, 0.1, self.weights2.shape)

# Agent Class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 100
        self.nn = NeuralNetwork()
        self.signal = 0
        self.memory = []  # Last 5 (action, reward) pairs

    def sense(self, grid, agents, N):
        resource_near = any(grid[max(0, self.x-3):min(N, self.x+4),
                                max(0, self.y-3):min(N, self.y+4)].flatten() > 0)
        agent_near = any(a for a in agents if a != self and
                         abs(a.x - self.x) <= 3 and abs(a.y - self.y) <= 3)
        return np.array([resource_near, agent_near, self.energy / 100])

    def act(self, grid, agents, N):
        inputs = self.sense(grid, agents, N)
        outputs = self.nn.forward(inputs)
        action = np.argmax(outputs)
        reward = 0

        if action == 0:  # Move
            self.x = (self.x + choice([-1, 0, 1])) % N
            self.y = (self.y + choice([-1, 0, 1])) % N
            self.energy -= 1
        elif action == 1 and grid[self.x, self.y] > 0:  # Collect
            reward = grid[self.x, self.y]
            self.energy += reward
            grid[self.x, self.y] = 0
        self.signal = 1 if outputs[2] > 0.5 else 0  # Communicate

        self.energy -= 1  # Base energy cost
        self.memory.append((action, reward))
        if len(self.memory) > 5:
            self.memory.pop(0)
        return reward

    def learn(self):
        if self.memory:
            avg_reward = sum(r for _, r in self.memory) / len(self.memory)
            if avg_reward > 0:
                self.nn.mutate()  # Positive reinforcement

    def reproduce(self):
        if self.energy > 150:
            child = Agent(self.x, self.y)
            child.nn = NeuralNetwork()
            child.nn.weights1 = self.nn.weights1.copy()
            child.nn.weights2 = self.nn.weights2.copy()
            child.nn.mutate()
            self.energy /= 2
            return child
        return None

# Simulation Update Function
def update(frame, img, grid, agents, N):
    # Resource regeneration
    for i in range(N):
        for j in range(N):
            if uniform(0, 1) < 0.01:
                grid[i, j] = min(10, grid[i, j] + 1)

    # Environmental event
    if frame % 100 == 0:
        if randint(0, 1):
            for _ in range(N * N // 10):
                grid[randint(0, N-1), randint(0, N-1)] += 5
        else:
            grid //= 2

    # Agent actions
    new_agents = []
    for agent in agents[:]:
        reward = agent.act(grid, agents, N)
        agent.learn()
        if agent.energy <= 0:
            agents.remove(agent)
        elif child := agent.reproduce():
            new_agents.append(child)

    agents.extend(new_agents)

    # Cooperation
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            if a1.signal == a2.signal and abs(a1.x - a2.x) <= 3 and abs(a1.y - a2.y) <= 3:
                avg_energy = (a1.energy + a2.energy) / 2
                a1.energy = a2.energy = avg_energy

    # Visualization
    display = grid.copy()
    for agent in agents:
        display[agent.x, agent.y] = 15  # Agents in white
    img.set_data(display)

    # Monitoring
    if frame % 100 == 0:
        print(f"Step {frame}: Population = {len(agents)}, Avg Energy = {np.mean([a.energy for a in agents]):.1f}")
    
    return img,

# Main Function
def main():
    N = 100
    grid = np.random.randint(0, 5, (N, N))
    agents = [Agent(randint(0, N-1), randint(0, N-1)) for _ in range(20)]

    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='viridis')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, agents, N),
                                  interval=50, save_count=200)

    plt.show()

if __name__ == "__main__":
    main()
