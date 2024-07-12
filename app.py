import tkinter as tk
import numpy as np
import random
import time

class GridGameGUI:
    def __init__(self, size=5, epochs=1000, target_threshold=1):
        self.size = size
        self.epochs = epochs
        self.target_reached_count = 0
        self.target_threshold = target_threshold
        self.agent_pos = [0, 0]
        self.obstacles = {'Instagram': [1, 1], 'Movies': [2, 2], 'YouTube': [3, 3]}  # Named obstacles
        self.target_pos = [size - 1, size - 1]
        self.cell_size = 60
        self.q_table = np.zeros((size * size, 4))  # Q-table for learning

        self.root = tk.Tk()
        self.root.title("Grid Game")
        self.canvas = tk.Canvas(self.root, width=self.size * self.cell_size,
                                height=self.size * self.cell_size + 30)
        self.canvas.pack()
        
        self.status_text = self.canvas.create_text(
            self.size * self.cell_size / 2, self.size * self.cell_size + 15,
            text="Starting training...", fill="black"
        )

        self.draw_grid()
        self.update_gui()

    def draw_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

    def update_gui(self):
        self.canvas.delete("agent")
        self.canvas.delete("obstacle")
        self.canvas.delete("target")

        for name, pos in self.obstacles.items():
            ox1, oy1, ox2, oy2 = self.get_cell_coords(pos)
            self.canvas.create_rectangle(ox1, oy1, ox2, oy2, fill="red", outline="black", tags="obstacle")
            self.canvas.create_text((ox1 + ox2) / 2, (oy1 + oy2) / 2, text=name, fill="white")

        if self.agent_pos == self.target_pos:
            tx1, ty1, tx2, ty2 = self.get_cell_coords(self.target_pos)
            self.canvas.create_rectangle(tx1, ty1, tx2, ty2, fill="purple", outline="black", tags="target")
            self.canvas.create_text((tx1 + tx2) / 2, (ty1 + ty2) / 2, text="Sleep", fill="white")
        else:
            tx1, ty1, tx2, ty2 = self.get_cell_coords(self.target_pos)
            self.canvas.create_rectangle(tx1, ty1, tx2, ty2, fill="green", outline="black", tags="target")
            self.canvas.create_text((tx1 + tx2) / 2, (ty1 + ty2) / 2, text="Sleep", fill="white")

        ax1, ay1, ax2, ay2 = self.get_cell_coords(self.agent_pos)
        self.canvas.create_rectangle(ax1, ay1, ax2, ay2, fill="blue", outline="black", tags="agent")
        self.canvas.create_text((ax1 + ax2) / 2, (ay1 + ay2) / 2, text="Rushi", fill="white")

    def get_cell_coords(self, pos):
        x1 = pos[1] * self.cell_size
        y1 = pos[0] * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return x1, y1, x2, y2

    def state_to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def choose_action(self, state_index, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state_index])

    def learn(self, old_state, new_state, action, reward, alpha=0.1, gamma=0.9):
        old_value = self.q_table[old_state, action]
        future_reward = np.max(self.q_table[new_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * future_reward)
        self.q_table[old_state, action] = new_value

    def move_agent(self, action):
        new_pos = self.agent_pos.copy()
        if action == 0 and new_pos[0] > 0:
            new_pos[0] -= 1
        elif action == 1 and new_pos[0] < self.size - 1:
            new_pos[0] += 1
        elif action == 2 and new_pos[1] > 0:
            new_pos[1] -= 1
        elif action == 3 and new_pos[1] < self.size - 1:
            new_pos[1] += 1

        if new_pos in [pos for name, pos in self.obstacles.items()]:
            new_pos = [0, 0]

        self.agent_pos = new_pos
        self.update_gui()

    def run_epoch(self, delay=0.1):
        self.agent_pos = [0, 0]
        state_index = self.state_to_index(self.agent_pos)
        done = False

        while not done:
            action = self.choose_action(state_index)
            old_state_index = state_index

            self.move_agent(action)
            state_index = self.state_to_index(self.agent_pos)

            reward = self.get_reward()
            done = self.is_done()
            self.learn(old_state_index, state_index, action, reward)

            action_text = ["Up", "Down", "Left", "Right"][action]
            mistake_text = "Hit obstacle" if self.agent_pos in [pos for name, pos in self.obstacles.items()] else ""
            self.canvas.itemconfig(
                self.status_text, 
                text=f"Action: {action_text}, Reward: {reward}, {mistake_text}"
            )

            time.sleep(delay)
            self.canvas.update()

            if self.agent_pos == self.target_pos:
                self.target_reached_count += 1
                if self.target_reached_count >= self.target_threshold:
                    self.canvas.create_text(
                        self.size * self.cell_size / 2, self.size * self.cell_size / 2, 
                        text="Rushi reached Sleep!", fill="purple", font=("Helvetica", 16)
                    )
                    return True
            else:
                self.target_reached_count = 0

        return False

    def get_reward(self):
        if self.agent_pos == self.target_pos:
            return 100
        elif self.agent_pos in [pos for name, pos in self.obstacles.items()]:
            return -100
        return -1

    def is_done(self):
        return self.agent_pos == self.target_pos

    def train(self):
        for epoch in range(self.epochs):
            if self.run_epoch(0.05):
                print(f"Training stopped early at epoch {epoch + 1}")
                break
            print(f"Epoch {epoch + 1}/{self.epochs} completed")

        # Save the Q-table model
        np.save("q_table.npy", self.q_table)
        print("Model saved.")

    def start(self):
        self.train()
        self.root.mainloop()

# Usage
game = GridGameGUI()
game.start()