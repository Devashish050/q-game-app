import streamlit as st
import numpy as np
import random
import time
import os  

class GridGame:
    def __init__(self, size=5, epochs=1000, target_threshold=1):
        self.size = size
        self.epochs = epochs
        self.target_reached_count = 0
        self.target_threshold = target_threshold
        self.agent_pos = [0, 0]
        self.obstacles = {'Instagram': [1, 1], 'Movies': [2, 2], 'YouTube': [3, 3]}  # Named obstacles
        self.target_pos = [size - 1, size - 1]
        self.q_table = np.zeros((size * size, 4))  # Q-table for learning

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

    def get_reward(self):
        if self.agent_pos == self.target_pos:
            return 100
        elif self.agent_pos in [pos for name, pos in self.obstacles.items()]:
            return -100
        return -1

    def is_done(self):
        return self.agent_pos == self.target_pos

    def run_epoch(self):
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

            if self.agent_pos == self.target_pos:
                self.target_reached_count += 1
                if self.target_reached_count >= self.target_threshold:
                    return True
            else:
                self.target_reached_count = 0

        return False

    def train(self):
        for epoch in range(self.epochs):
            if self.run_epoch():
                print(f"Training stopped early at epoch {epoch + 1}")
                break
            print(f"Epoch {epoch + 1}/{self.epochs} completed")

        # Save the Q-table model
        np.save("q_table.npy", self.q_table)
        print("Model saved.")

def main():
    st.title("Grid Game")

    size = st.sidebar.slider("Grid Size", 3, 10, 5)
    epochs = st.sidebar.slider("Epochs", 100, 10000, 1000, 100)
    target_threshold = st.sidebar.slider("Target Threshold", 1, 10, 1)

    if st.sidebar.button("Start Training"):
        game = GridGame(size=size, epochs=epochs, target_threshold=target_threshold)
        game.train()
        st.success("Training completed. Model saved as q_table.npy.")

    st.sidebar.write("Q-Table (First 10 entries):")
    if "q_table.npy" in os.listdir():
        q_table = np.load("q_table.npy")
        st.sidebar.write(q_table[:10])
    else:
        st.sidebar.write("No Q-Table found. Please run the training first.")

if __name__ == "__main__":
    main()
