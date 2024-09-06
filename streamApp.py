import streamlit as st
import numpy as np
import random
import time
import uuid

# Grid Game Class
class GridGameStreamlit:
    def __init__(self, size=5, epochs=1000, target_threshold=1):
        self.size = size
        self.epochs = epochs
        self.target_reached_count = 0
        self.target_threshold = target_threshold
        self.agent_pos = [0, 0]
        self.obstacles = {'Instagram': [1, 1], 'Movies': [2, 2], 'YouTube': [3, 3]}  # Named obstacles
        self.target_pos = [size - 1, size - 1]
        self.q_table = np.zeros((size * size, 4))  # Q-table for learning

        # Initialize session state for storing positions and other game state
        if 'agent_pos' not in st.session_state:
            st.session_state['agent_pos'] = self.agent_pos
        if 'target_reached_count' not in st.session_state:
            st.session_state['target_reached_count'] = self.target_reached_count

    def draw_grid(self):
        st.write(f"**Target Position:** {self.target_pos}")
        st.write(f"**Obstacles:** {self.obstacles}")

        # Display the grid with agent, obstacles, and target
        unique_id = str(uuid.uuid4())  # Generate a unique identifier for each draw
        for i in range(self.size):
            cols = st.columns(self.size)
            for j in range(self.size):
                pos = [i, j]
                key_prefix = f"pos_{i}_{j}_{unique_id}_"

                if pos == st.session_state['agent_pos']:
                    cols[j].button("Rushi", key=f"{key_prefix}agent", disabled=True)
                elif pos == self.target_pos:
                    cols[j].button("Sleep", key=f"{key_prefix}target", disabled=True)
                elif pos in self.obstacles.values():
                    # Find the obstacle name based on the position
                    obstacle_name = next(name for name, position in self.obstacles.items() if position == pos)
                    cols[j].button(obstacle_name, key=f"{key_prefix}obstacle", disabled=True)
                else:
                    cols[j].button("", key=f"{key_prefix}empty", disabled=True)

    def state_to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def choose_action(self, state_index, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)  # Random action (0: up, 1: down, 2: left, 3: right)
        else:
            return np.argmax(self.q_table[state_index])

    def learn(self, old_state, new_state, action, reward, alpha=0.1, gamma=0.9):
        old_value = self.q_table[old_state, action]
        future_reward = np.max(self.q_table[new_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * future_reward)
        self.q_table[old_state, action] = new_value

    def move_agent(self, action):
        new_pos = st.session_state['agent_pos'].copy()
        if action == 0 and new_pos[0] > 0:  # Move up
            new_pos[0] -= 1
        elif action == 1 and new_pos[0] < self.size - 1:  # Move down
            new_pos[0] += 1
        elif action == 2 and new_pos[1] > 0:  # Move left
            new_pos[1] -= 1
        elif action == 3 and new_pos[1] < self.size - 1:  # Move right
            new_pos[1] += 1

        # Check if the agent hit an obstacle
        if new_pos in [pos for name, pos in self.obstacles.items()]:
            st.write("**Agent hit an obstacle and is reset to the starting position!**")
            new_pos = [0, 0]

        st.session_state['agent_pos'] = new_pos
        self.draw_grid()

    def run_epoch(self):
        st.session_state['agent_pos'] = [0, 0]
        state_index = self.state_to_index(st.session_state['agent_pos'])
        done = False

        while not done:
            action = self.choose_action(state_index)
            old_state_index = state_index

            self.move_agent(action)
            state_index = self.state_to_index(st.session_state['agent_pos'])

            reward = self.get_reward()
            done = self.is_done()
            self.learn(old_state_index, state_index, action, reward)

            action_text = ["Up", "Down", "Left", "Right"][action]
            st.write(f"Action: **{action_text}**, Reward: **{reward}**")

            time.sleep(0.5)  # Delay to simulate the game process

            if st.session_state['agent_pos'] == self.target_pos:
                st.session_state['target_reached_count'] += 1
                if st.session_state['target_reached_count'] >= self.target_threshold:
                    st.success("**Rushi reached Sleep! Training complete.**")
                    return True
            else:
                st.session_state['target_reached_count'] = 0

        return False

    def get_reward(self):
        if st.session_state['agent_pos'] == self.target_pos:
            return 100
        elif st.session_state['agent_pos'] in [pos for name, pos in self.obstacles.items()]:
            return -100
        return -1

    def is_done(self):
        return st.session_state['agent_pos'] == self.target_pos

    def start(self):
        for epoch in range(self.epochs):
            if self.run_epoch():
                st.write(f"Training stopped early at epoch {epoch + 1}")
                break
            st.write(f"Epoch {epoch + 1}/{self.epochs} completed")

        # Save the Q-table model
        np.save("q_table_streamlit.npy", self.q_table)
        st.write("Model saved.")

# Streamlit UI
st.title("Grid Game - Streamlit Version")

size = st.slider("Grid Size", min_value=3, max_value=10, value=5)
epochs = st.slider("Epochs", min_value=100, max_value=1000, value=500)
target_threshold = st.slider("Target Threshold", min_value=1, max_value=5, value=1)

game = GridGameStreamlit(size=size, epochs=epochs, target_threshold=target_threshold)

if st.button("Start Training"):
    game.start()
else:
    game.draw_grid()
