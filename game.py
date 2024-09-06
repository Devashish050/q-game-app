import streamlit as st
import numpy as np
import random
import time
from PIL import Image, ImageDraw, ImageFont

# Grid Game Class
class GridGameStreamlit:
    def __init__(self, size=5, epochs=1000, target_threshold=1, cell_size=60):
        self.size = size
        self.epochs = epochs
        self.target_reached_count = 0
        self.target_threshold = target_threshold
        self.cell_size = cell_size  # Size of each cell in pixels
        self.agent_pos = [0, 0]
        self.obstacles = {'Instagram': [1, 1], 'Movies': [2, 2], 'YouTube': [3, 3]}  # Named obstacles
        self.target_pos = [size - 1, size - 1]
        self.q_table = np.zeros((size * size, 4))  # Q-table for learning

        # Initialize session state for storing positions and other game state
        if 'agent_pos' not in st.session_state:
            st.session_state['agent_pos'] = self.agent_pos
        if 'target_reached_count' not in st.session_state:
            st.session_state['target_reached_count'] = self.target_reached_count
        if 'epoch' not in st.session_state:
            st.session_state['epoch'] = 0

    def draw_grid(self):
        # Create a blank white image
        img_size = self.size * self.cell_size
        image = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(image)

        # Define colors
        agent_color = 'blue'
        target_color = 'green'
        obstacle_color = 'red'
        grid_color = 'black'

        # Optional: Load a font
        try:
            font = ImageFont.truetype("arial.ttf", int(self.cell_size / 4))
        except:
            font = ImageFont.load_default()

        # Draw grid lines
        for i in range(self.size + 1):
            # Horizontal lines
            draw.line([(0, i * self.cell_size), (img_size, i * self.cell_size)], fill=grid_color)
            # Vertical lines
            draw.line([(i * self.cell_size, 0), (i * self.cell_size, img_size)], fill=grid_color)

        # Draw obstacles
        for name, pos in self.obstacles.items():
            top_left = (pos[1] * self.cell_size, pos[0] * self.cell_size)
            bottom_right = ((pos[1] + 1) * self.cell_size, (pos[0] + 1) * self.cell_size)
            draw.rectangle([top_left, bottom_right], fill=obstacle_color)
            # Add obstacle name
            text_bbox = draw.textbbox(top_left, name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (
                top_left[0] + (self.cell_size - text_width) / 2,
                top_left[1] + (self.cell_size - text_height) / 2
            )
            draw.text(text_position, name, fill='white', font=font)

        # Draw target
        target = self.target_pos
        top_left = (target[1] * self.cell_size, target[0] * self.cell_size)
        bottom_right = ((target[1] + 1) * self.cell_size, (target[0] + 1) * self.cell_size)
        draw.rectangle([top_left, bottom_right], fill=target_color)
        # Add 'Sleep' label
        text = "Sleep"
        text_bbox = draw.textbbox(top_left, text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (
            top_left[0] + (self.cell_size - text_width) / 2,
            top_left[1] + (self.cell_size - text_height) / 2
        )
        draw.text(text_position, text, fill='white', font=font)

        # Draw agent
        agent = st.session_state['agent_pos']
        top_left = (agent[1] * self.cell_size, agent[0] * self.cell_size)
        bottom_right = ((agent[1] + 1) * self.cell_size, (agent[0] + 1) * self.cell_size)
        draw.ellipse([top_left, bottom_right], fill=agent_color)
        # Add 'Agent RL' label
        text = "Agent RL"
        text_bbox = draw.textbbox(top_left, text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (
            top_left[0] + (self.cell_size - text_width) / 2,
            top_left[1] + (self.cell_size - text_height) / 2
        )
        draw.text(text_position, text, fill='white', font=font)

        return image

    def run_epoch(self, grid_placeholder):
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
            st.write(f"**Action:** {action_text} | **Reward:** {reward}")

            # Update the grid image
            grid_image = self.draw_grid()
            grid_placeholder.image(grid_image, use_column_width=True)

            time.sleep(1)  # Delay to simulate the game process

            if st.session_state['agent_pos'] == self.target_pos:
                st.session_state['target_reached_count'] += 1
                if st.session_state['target_reached_count'] >= self.target_threshold:
                    st.success("**Agent RL reached Sleep! Training complete.**")
                    return True
            else:
                st.session_state['target_reached_count'] = 0

        return False

    def state_to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def choose_action(self, state_index, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)  # Random action (0: up, 1: down, 2: left, 3: right)
        else:
            return np.argmax(self.q_table[state_index])

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

    def learn(self, old_state, new_state, action, reward, alpha=0.1, gamma=0.9):
        old_value = self.q_table[old_state, action]
        future_reward = np.max(self.q_table[new_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * future_reward)
        self.q_table[old_state, action] = new_value

    def get_reward(self):
        if st.session_state['agent_pos'] == self.target_pos:
            return 100
        elif st.session_state['agent_pos'] in [pos for name, pos in self.obstacles.items()]:
            return -100
        return -1

    def is_done(self):
        return st.session_state['agent_pos'] == self.target_pos

    def start(self, grid_placeholder):
        for epoch in range(self.epochs):
            st.session_state['epoch'] = epoch + 1
            if self.run_epoch(grid_placeholder):
                st.write(f"Training stopped early at epoch {epoch + 1}")
                break
            st.write(f"Epoch {epoch + 1}/{self.epochs} completed")

        # Save the Q-table model
        np.save("q_table_streamlit.npy", self.q_table)
        st.write("**Model saved.**")


# Streamlit UI
st.set_page_config(page_title="Grid Game - Streamlit Version", layout="centered")
st.title("🕹️ Grid Game - Streamlit Version")

# Sidebar for controls
st.sidebar.header("Game Controls")
size = st.sidebar.slider("Grid Size", min_value=3, max_value=10, value=5)
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=1000, value=500)
target_threshold = st.sidebar.slider("Target Threshold", min_value=1, max_value=5, value=1)

game = GridGameStreamlit(size=size, epochs=epochs, target_threshold=target_threshold)

# Placeholder for the grid image
grid_placeholder = st.empty()

# Initialize the grid image
initial_grid = game.draw_grid()
grid_placeholder.image(initial_grid, use_column_width=True)

# Button to start training
if st.button("🏁 Start Training"):
    game.start(grid_placeholder)



# import streamlit as st
# import numpy as np
# import random
# import time
# from PIL import Image, ImageDraw, ImageFont

# # Grid Game Class
# class GridGameStreamlit:
#     def __init__(self, size=5, epochs=1000, target_threshold=1, cell_size=60):
#         self.size = size
#         self.epochs = epochs
#         self.target_reached_count = 0
#         self.target_threshold = target_threshold
#         self.cell_size = cell_size  # Size of each cell in pixels
#         self.agent_pos = [0, 0]
#         self.obstacles = {'Instagram': [1, 1], 'Movies': [2, 2], 'YouTube': [3, 3]}  # Named obstacles
#         self.target_pos = [size - 1, size - 1]
#         self.q_table = np.zeros((size * size, 4))  # Q-table for learning

#         # Initialize session state for storing positions and other game state
#         if 'agent_pos' not in st.session_state:
#             st.session_state['agent_pos'] = self.agent_pos
#         if 'target_reached_count' not in st.session_state:
#             st.session_state['target_reached_count'] = self.target_reached_count
#         if 'epoch' not in st.session_state:
#             st.session_state['epoch'] = 0

#     def draw_grid(self):
#         # Create a blank white image
#         img_size = self.size * self.cell_size
#         image = Image.new('RGB', (img_size, img_size), color='white')
#         draw = ImageDraw.Draw(image)

#         # Define colors
#         agent_color = 'blue'
#         target_color = 'green'
#         obstacle_color = 'red'
#         grid_color = 'black'

#         # Optional: Load a font
#         try:
#             font = ImageFont.truetype("arial.ttf", int(self.cell_size / 4))
#         except:
#             font = ImageFont.load_default()

#         # Draw grid lines
#         for i in range(self.size + 1):
#             # Horizontal lines
#             draw.line([(0, i * self.cell_size), (img_size, i * self.cell_size)], fill=grid_color)
#             # Vertical lines
#             draw.line([(i * self.cell_size, 0), (i * self.cell_size, img_size)], fill=grid_color)

#         # Draw obstacles
#         for name, pos in self.obstacles.items():
#             top_left = (pos[1] * self.cell_size, pos[0] * self.cell_size)
#             bottom_right = ((pos[1] + 1) * self.cell_size, (pos[0] + 1) * self.cell_size)
#             draw.rectangle([top_left, bottom_right], fill=obstacle_color)
#             # Add obstacle name
#             text_bbox = draw.textbbox(top_left, name, font=font)
#             text_width = text_bbox[2] - text_bbox[0]
#             text_height = text_bbox[3] - text_bbox[1]
#             text_position = (
#                 top_left[0] + (self.cell_size - text_width) / 2,
#                 top_left[1] + (self.cell_size - text_height) / 2
#             )
#             draw.text(text_position, name, fill='white', font=font)

#         # Draw target
#         target = self.target_pos
#         top_left = (target[1] * self.cell_size, target[0] * self.cell_size)
#         bottom_right = ((target[1] + 1) * self.cell_size, (target[0] + 1) * self.cell_size)
#         draw.rectangle([top_left, bottom_right], fill=target_color)
#         # Add 'Sleep' label
#         text = "Sleep"
#         text_bbox = draw.textbbox(top_left, text, font=font)
#         text_width = text_bbox[2] - text_bbox[0]
#         text_height = text_bbox[3] - text_bbox[1]
#         text_position = (
#             top_left[0] + (self.cell_size - text_width) / 2,
#             top_left[1] + (self.cell_size - text_height) / 2
#         )
#         draw.text(text_position, text, fill='white', font=font)

#         # Draw agent
#         agent = st.session_state['agent_pos']
#         top_left = (agent[1] * self.cell_size, agent[0] * self.cell_size)
#         bottom_right = ((agent[1] + 1) * self.cell_size, (agent[0] + 1) * self.cell_size)
#         draw.ellipse([top_left, bottom_right], fill=agent_color)
#         # Add 'Rushi' label
#         text = "Rushi"
#         text_bbox = draw.textbbox(top_left, text, font=font)
#         text_width = text_bbox[2] - text_bbox[0]
#         text_height = text_bbox[3] - text_bbox[1]
#         text_position = (
#             top_left[0] + (self.cell_size - text_width) / 2,
#             top_left[1] + (self.cell_size - text_height) / 2
#         )
#         draw.text(text_position, text, fill='white', font=font)

#         return image

#     # (Rest of the class remains unchanged...)

# # Streamlit UI
# st.set_page_config(page_title="Grid Game - Streamlit Version", layout="centered")
# st.title("🕹️ Grid Game - Streamlit Version")

# # Sidebar for controls
# st.sidebar.header("Game Controls")
# size = st.sidebar.slider("Grid Size", min_value=3, max_value=10, value=5)
# epochs = st.sidebar.slider("Epochs", min_value=100, max_value=1000, value=500)
# target_threshold = st.sidebar.slider("Target Threshold", min_value=1, max_value=5, value=1)

# game = GridGameStreamlit(size=size, epochs=epochs, target_threshold=target_threshold)

# # Placeholder for the grid image
# grid_placeholder = st.empty()

# # Initialize the grid image
# initial_grid = game.draw_grid()
# grid_placeholder.image(initial_grid, use_column_width=True)

# # Button to start training
# if st.button("🏁 Start Training"):
#     game.start(grid_placeholder)
