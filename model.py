import torch
import torch.nn as nn
import torch.optim as optim # provides optimization algorithms for training neural networks
import torch.nn.functional as F # contains various activation functions and loss functions
import os # used here for file operations and path handling

# our model is a feed forward neural network
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # 1st layer, input is input_size and gives output of hidden_size
        self.linear2 = nn.Linear(hidden_size, output_size) # 2nd layer

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x # it returns the final output tensor, which contains the Q-values

    def save(self, file_name='model.pth'): # a file where the model's parameters will be saved
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): # if path doesn't exist, it creates a new
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # saved in a dictionery


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # initializes an Adam optimizer for training the neural network
        self.criterion = nn.MSELoss()  # MSE is mean square error 

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float) # converts into a pytorch tensor of type float
        next_state = torch.tensor(next_state, dtype=torch.float) # same
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) -> (batch size, dimensionality of the input data)

        if len(state.shape) == 1: # if it is 1D i.e, represents the single sample
            # (1, x)
            state = torch.unsqueeze(state, 0) # converts it into a 2D tensor
            next_state = torch.unsqueeze(next_state, 0) # same
            action = torch.unsqueeze(action, 0) # same
            reward = torch.unsqueeze(reward, 0) # same
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                # updates the Q-values based on Bellman equation
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad() # clears gradients, and prepares for new gradients
        loss = self.criterion(target, pred) # using MSE function, it calculates loss b/w target and predicted values
        loss.backward()

        self.optimizer.step() #  updates the model's weights and biases based on the computed gradients