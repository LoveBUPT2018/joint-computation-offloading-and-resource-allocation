# -*- coding: utf-8 -*-
"""
Author：fenghao
Date：2020.2.14
Location：BUPT
"""

import copy
import numpy as np
from numpy import pi
from collections import namedtuple
from random import random, uniform, choice, randrange, sample
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from scenario import Scenario, MEC

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple

class ReplayMemory(object):    # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DNN(nn.Module):  # Define a deep neural network

    def __init__(self, opt, sce, scenario):  # Define the layers of the fully-connected hidden network
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(opt.nagents, 64)
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, scenario.MEC_Number() * sce.nRB)
		
    def forward(self, state):  # Define the neural network forward function
        x1 = F.relu(self.input_layer(state))
        x2 = F.relu(self.middle1_layer(x1))
        x3 = F.relu(self.middle2_layer(x2))
        out = self.output_layer(x3)
        return out
        

class Agent:  # Define the agent (UE)
    
    def __init__(self, opt, sce, scenario, index, device):  # Initialize the agent (UE)
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        self.location = self.Set_Location(scenario)
        self.memory = ReplayMemory(opt.capacity)
        self.model_policy = DNN(opt, sce, scenario)
        self.model_target = DNN(opt, sce, scenario)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_policy.parameters(), lr=opt.learningrate, momentum=opt.momentum)

    def Set_Location(self, scenario):  # Initialize the location of the agent
        Loc_MMEC, _ , _ = scenario.MEC_Location()
        Loc_agent = np.zeros(2)
        LocM = choice(Loc_MMEC)
        r = self.sce.rMMEC*random()
        theta = uniform(-pi,pi)
        Loc_agent[0] = LocM[0] + r*np.cos(theta)
        Loc_agent[1] = LocM[1] + r*np.sin(theta) 
        return Loc_agent
    
    def Get_Location(self):
        return self.location
     
    def Select_Action(self, state, scenario, eps_threshold):   # Select action for a user based on the network state
        L = scenario.MEC_Number()  # The total number of MECs
        K = self.sce.nRB  # The total number of RBs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        sample = random()       
        if sample < eps_threshold:  # epsilon-greeedy policy
            with torch.no_grad():
                Q_value = self.model_policy(state)   # Get the Q_value from DNN
                action = Q_value.max(0)[1].view(1,1)
        else:           
            action = torch.tensor([[randrange(L*K)]], dtype=torch.long)
        return action      
        		
    def Get_Reward(self, action, action_i, state, scenario):  # Get reward for the state-action pair
        MEC = scenario.Get_MECs()
        L = scenario.MEC_Number()  # The total number of MECs
        K = self.sce.nRB  # The total number of RBs 

        MEC_selected = action_i // K
        Ch_selected = action_i % K  # Translate to the selected MEC and RB based on the selected action index
        Loc_diff = MEC[MEC_selected].Get_Location() - self.location
        distance = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))  # Calculate the distance between MEC and UE
        Rx_power = MEC[MEC_selected].Receive_Power(distance)  # Calculate the received power
        
        if Rx_power == 0.0:
            reward = self.sce.negative_cost  # Out of range of the selected MEC, thus obtain a negative reward
            QoS = 0  # Definitely, QoS cannot be satisfied
        else:                    # If inside the coverage, then we will calculate the reward value
            Interference = 0.0
            for i in range(self.opt.nagents):   # Obtain interference on the same RB
                MEC_select_i = action[i] // K
                Ch_select_i = action[i] % K   # The choice of other users
                if Ch_select_i == Ch_selected:  # Calculate the interference on the same RB
                    Loc_diff_i = MEC[MEC_select_i].Get_Location() - self.location
                    distance_i = np.sqrt((Loc_diff_i[0]**2 + Loc_diff_i[1]**2))
                    Rx_power_i = MEC[MEC_select_i].Receive_Power(distance_i)
                    Interference += Rx_power_i   # Sum all the interference
            Interference -= Rx_power  # Remove the received power from interference
            Noise = 10**((self.sce.N0)/10)*self.sce.BW  # Calculate the noise
            SINR = Rx_power/(Interference + Noise)  # Calculate the SINR      
            if SINR >= 10**(self.sce.QoS_thr/10):
                QoS = 1
                reward = 1
            else:
                QoS = 0   
                reward = self.sce.negative_cost
            """Rate = self.sce.BW * np.log2(1 + SINR) / (10**6)      # Calculate the rate of UE 
            profit = self.sce.profit * Rate
            Tx_power_dBm = MEC[MEC_selected].Transmit_Power_dBm()   # Calculate the transmit power of the selected MEC
            cost = self.sce.power_cost * Tx_power_dBm + self.sce.action_cost  # Calculate the total cost
            reward = profit - cost """
        reward = torch.tensor([reward])
        return QoS, reward
    
    def Save_Transition(self, state, action, next_state, reward, scenario):  # Store a transition
        L = scenario.MEC_Number()     # The total number of MECs
        K = self.sce.nRB        # The total number of RBs
        action = torch.tensor([[action]])
        reward = torch.tensor([reward])
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        self.memory.Push(state, action, next_state, reward)
    
    def Target_Update(self):  # Update the parameters of the target network
        self.model_target.load_state_dict(self.model_policy.state_dict())
            
    def Optimize_Model(self):
        if len(self.memory) < self.opt.batch_size:
            return
        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.model_policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.opt.batch_size)
        
        next_action_batch = torch.unsqueeze(self.model_policy(non_final_next_states).max(1)[1], 1)
        next_state_values = self.model_target(non_final_next_states).gather(1, next_action_batch) 
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch.unsqueeze(1) 
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # DDPG
        """
        next_state_values[non_final_mask] = self.model_target(non_final_next_states).max(1)[0].detach()  # DDPG
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        """
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

            
            

            
            
            
            
            
            
            
            
            
            
            
            


        
        
        
