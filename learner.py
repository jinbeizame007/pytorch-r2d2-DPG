import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dm_control import suite
from collections import deque
from PIL import Image
import random
import gym
import os
from copy import deepcopy
from time import time, sleep

from replay_memory import ReplayMemory, LearnerReplayMemory
from models import ActorNet, CriticNet
from utils import soft_update, get_obs, calc_priority, invertical_vf

def learner_process(n_actors):
    learner = Learner(n_actors)
    learner.run()

class Learner:
    def __init__(self, n_actors):
        self.env = suite.load(domain_name="walker", task_name="run")
        self.n_actions = self.env.action_spec().shape[0]
        self.obs_size = get_obs(self.env.reset().observation).shape[1]

        self.n_actors = n_actors
        self.burn_in_length = 20 # 40-80
        self.learning_length = 40
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = 5
        self.memory_sequence_size = 5000000
        self.batch_size = 32
        self.memory = LearnerReplayMemory(memory_sequence_size=self.memory_sequence_size, batch_size=self.batch_size)

        self.model_path = './model_data/'
        self.memory_path = './memory_data/'
        self.actor = ActorNet(self.obs_size, self.n_actions, 0).cuda()
        self.target_actor = deepcopy(self.actor).eval()
        self.critic = CriticNet(self.obs_size, self.n_actions, 0).cuda()
        self.target_critic = deepcopy(self.critic).eval()
        self.model_save_interval = 50 # 50
        self.memory_update_interval = 50 # 50
        self.target_update_inverval = 500 # 100

        self.gamma = 0.997
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_criterion = nn.MSELoss()
        self.save_model()
    
    def save_model(self):
        model_dict = {'actor': self.actor.state_dict(),
                      'target_actor': self.target_actor.state_dict(),
                      'critic': self.critic.state_dict(),
                      'target_critic': self.target_critic.state_dict()}
        torch.save(model_dict, self.model_path + 'model.pt')
    
    def update_target_model(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def run(self):
        # memory not enough
        while self.memory.sequence_counter < self.batch_size * 100:
            for i in range(self.n_actors):
                is_memory = os.path.isfile(self.memory_path + '/memory{}.pt'.format(i))
                if is_memory:
                    self.memory.load(i)
                sleep(0.1)
            print('learner memory sequence size:', self.memory.sequence_counter)
        
        step = 0
        while True:
            if step % 100 == 0:
                print('learning step:', step)
            start = time()
            step += 1

            episode_index, sequence_index, obs_seq, action_seq, reward_seq, terminal_seq, a_state, ta_state, c_state, tc_state = self.memory.sample()

            self.actor.set_state(a_state[0], a_state[1])
            self.target_actor.set_state(ta_state[0], ta_state[1])
            self.critic.set_state(c_state[0], c_state[1])
            self.target_critic.set_state(tc_state[0], tc_state[1])

            ### burn-in step ###
            _ = [self.actor(obs) for obs in obs_seq[0:self.burn_in_length]]
            _ = [self.critic(obs, action) for obs, action in zip(obs_seq[0:self.burn_in_length], action_seq[0:self.burn_in_length])]
            _ = [self.target_actor(obs) for obs in obs_seq[0:self.burn_in_length+self.n_step]]
            _ = [self.target_critic(obs, action) for obs, action in zip(obs_seq[0:self.burn_in_length+self.n_step], action_seq[0:self.burn_in_length+self.n_step])]
            
            ### learning steps ###

            # update ciritic
            q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions).cuda()
            target_q_value = torch.zeros(self.learning_length * self.batch_size, self.n_actions).cuda()
            for i in range(self.learning_length):
                obs_i = self.burn_in_length + i
                next_obs_i = self.burn_in_length + i + self.n_step
                q_value[i*self.batch_size: (i+1)*self.batch_size] = self.critic(obs_seq[obs_i], action_seq[obs_i])
                next_q_value = self.target_critic(obs_seq[next_obs_i], self.target_actor(obs_seq[next_obs_i]))
                target_q_val = reward_seq[obs_i] + (self.gamma ** self.n_step) * (1. - terminal_seq[next_obs_i-1]) * next_q_value
                target_q_val = invertical_vf(target_q_val)
                target_q_value[i*self.batch_size: (i+1)*self.batch_size] = target_q_val
            
            critic_loss = self.actor_criterion(q_value, target_q_value.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # update actor
            self.actor.reset_state()
            self.critic.reset_state()
            actor_loss = torch.zeros(self.learning_length * self.batch_size, self.n_actions).cuda()
            for i in range(self.learning_length):
                obs_i = i + self.burn_in_length
                action = self.actor(obs_seq[obs_i])
                actor_loss[i*self.batch_size: (i+1)*self.batch_size] = -self.critic(obs_seq[obs_i], self.actor(obs_seq[obs_i]))
            actor_loss = actor_loss.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            if step % self.target_update_inverval == 0:
                self.update_target_model()

            # calc priority
            average_td_loss = np.mean((q_value - target_q_value).detach().cpu().numpy() ** 2., axis = 1)
            for i in range(len(episode_index)):
                td = average_td_loss[i: -1: self.batch_size]
                self.memory.priority[episode_index[i]][sequence_index[i]] = calc_priority(td)
                self.memory.total_priority[episode_index[i]] = sum(self.memory.priority[episode_index[i]])

            if step % self.model_save_interval == 0:
                self.save_model()

            if step % self.memory_update_interval == 0:
                for i in range(self.n_actors):
                    is_memory = os.path.isfile(self.memory_path + '/memory{}.pt'.format(i))
                    if is_memory:
                        self.memory.load(i)
                    sleep(0.1)

            self.actor.reset_state()
            self.target_actor.reset_state()
            self.critic.reset_state()
            self.target_critic.reset_state()
