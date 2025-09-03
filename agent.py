import torch
import random
import numpy as np
from collections import deque # a double ended queue for storing replay memories
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from game_plot import plot

MAX_MEMORY = 100_000 # replay memory, here it is upto 100,000 replays or transitions
BATCH_SIZE = 1000 # no. of replays sampled for each training set
LR = 0.001 # the rate at which the weights are updated during training

class Agent:

    def __init__(self):  # constructor
        self.n_games = 0 # initialises the no. of games to 0
        self.epsilon = 0 #  Epsilon (ε) represents the exploration vs. exploitation trade-off 
        # in reinforcement learning. An epsilon-greedy strategy is used to decide whether to explore new actions
        # (with probability ε) or exploit the current best-known action (with probability 1-ε).
        self.gamma = 0.92 # Gamma (γ) is the discount rate used in the Q-Learning algorithm. It determines
        # the importance of future rewards. A value of xx is used, which means future rewards are discounted
        # by 10% per time step.
        self.memory = deque(maxlen=MAX_MEMORY) # popleft(), max memory it can store is up to MAX_MEMORY, when the
        # new experiences are added, old ones are removed if they exceed the memory
        self.model = Linear_QNet(11, 256, 3) # 3 arguments, input size, hidden layer size, and output size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # responsible for training the QNetwork

        # Q values are updated using the Bellman equation:
        # Q(s, a) = Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]
        # The goal of Q-Learning is to learn the optimal Q-values

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # this returns the state as a NumPy array of integers. 
        # This array will be used as input to the neural network.

    def remember(self, state, action, reward, next_state, done): # It appends these arguments as tuple
        # to the agent's replay memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # it randomly samples a batch of experiences 
            # from the replay memory. If not, it uses the entire memory.
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # this method is called with these lists of experiences to update the Q-network's 
        # weights based on the Q-learning algorithm.

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # New formula: Epsilon will decrease but never go below 5%
        self.epsilon = max(0.05, 0.5 - self.n_games / 150)
        
        final_move = [0,0,0]
        # Use a float for the random check now
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent() # Create an instance of the Agent class
    game = SnakeGameAI() # Create an instance of the SnakeGameAI class

    # infinite training loop
    while True:
        # current state of the game
        state_old = agent.get_state(game)

        # agent's action based on the current state
        final_move = agent.get_action(state_old)

        # Perform the selected action in the game and get the new state, reward, and whether the game is done
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # if finished, reset the game environment
            game.reset()
            agent.n_games += 1 # incrementing the no. of games played by the agent
            agent.train_long_memory()

            if score > record: # update the record if the current score is higher
                record = score
                agent.model.save() # save the model with the best performance

            print('Game:', agent.n_games, ',Score:', score, ',Record:', record)

            plot_scores.append(score) # append the current score to the list
            total_score += score
            mean_score = total_score / agent.n_games # calculate the mean score
            plot_mean_scores.append(mean_score) # append the mean score to the list
            plot(plot_scores, plot_mean_scores) # plot

# It is a conditional statement that checks whether the current script is being run as the main program.
if __name__ == '__main__':
    agent = Agent()
    train()

    agent.model.save('final_model.pth')  # Save the final trained model