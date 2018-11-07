import torch
import numpy as np
from dm_control import suite
from collections import deque
from PIL import Image
import random
import gym
import os
from time import sleep, time
from copy import deepcopy

from replay_memory import ReplayMemory
from models import ActorNet, CriticNet
from utils import soft_update, get_obs, calc_priority, invertical_vf

def actor_process(actor_id):
    actor = Actor(actor_id)
    actor.run()

class Actor:
    def __init__(self, actor_id):
        self.env = suite.load(domain_name="walker", task_name="run")
        self.action_size = self.env.action_spec().shape[0]
        self.obs_size = get_obs(self.env.reset().observation).shape[1]

        self.actor_id = actor_id
        self.burn_in_length = 20 # 40-80
        self.learning_length = 40
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = 5
        self.sequence = []
        self.recurrent_state = []
        self.priority = []
        self.td_loss = deque(maxlen=self.learning_length)
        self.memory_sequence_size = 1000
        self.memory = ReplayMemory(memory_sequence_size=self.memory_sequence_size)
        self.memory_save_interval = 3

        self.gamma = 0.997
        self.actor_parameter_update_interval = 500
        self.model_path = './model_data/'
        self.actor = ActorNet(self.obs_size, self.action_size, cuda_id=self.actor_id%2+1).cuda(self.actor_id%2+1).eval()
        self.target_actor = deepcopy(self.actor)
        self.critic = CriticNet(self.obs_size, self.action_size, cuda_id=self.actor_id%2+1).cuda(self.actor_id%2+1).eval()
        self.target_critic = deepcopy(self.critic)
        self.load_model()
        self.epsilon = 1 
        self.last_obs = None
    
    def load_model(self):
        if os.path.isfile(self.model_path + 'model.pt'):
            while True:
                try:
                    # TODO: Delete
                    self.actor = ActorNet(self.obs_size, self.action_size, self.actor_id%2+1).cuda().eval()
                    self.target_actor = deepcopy(self.actor)
                    self.critic = CriticNet(self.obs_size, self.action_size, self.actor_id%2+1).cuda().eval()
                    self.target_critic = deepcopy(self.critic)
                    #model_dict = torch.load(self.model_path + 'model.pt', map_location={'cuda:0':'cuda:{}'.format(self.actor_id%2+1)})
                    model_dict = torch.load(self.model_path + 'model.pt')
                    self.actor.load_state_dict(model_dict['actor'])
                    self.target_actor.load_state_dict(model_dict['target_actor'])
                    self.critic.load_state_dict(model_dict['critic'])
                    self.target_critic.load_state_dict(model_dict['target_critic'])
                    self.actor.cuda(self.actor_id%2+1)
                    self.target_actor.cuda(self.actor_id%2+1)
                    self.critic.cuda(self.actor_id%2+1)
                    self.target_critic.cuda(self.actor_id%2+1)
                except:
                    sleep(np.random.rand() * 5 + 2)
                else:
                    break

    def calc_nstep_reward(self):
        for i in range(len(self.sequence) - self.n_step):
            self.sequence[i][2][0] = sum([self.sequence[i+j][2][0] * (self.gamma ** j) for j in range(self.n_step)])

    def calc_priorities(self):
        self.actor.reset_state()
        self.critic.reset_state()
        self.target_actor.reset_state()
        self.target_critic.reset_state()
        self.td_loss = deque(maxlen=self.learning_length)
        self.priority = []

        for i in range(self.n_step):
            next_obs = torch.from_numpy(self.sequence[i][0]).cuda(self.actor_id%2+1).unsqueeze(0)
            next_action = self.target_actor(next_obs)
            next_q_value = self.target_critic(next_obs, next_action).detach().cpu().numpy()

        for i in range(len(self.sequence) - self.n_step):
            obs = torch.from_numpy(self.sequence[i][0]).cuda(self.actor_id%2+1).unsqueeze(0)
            action = torch.from_numpy(self.sequence[i][1]).cuda(self.actor_id%2+1).unsqueeze(0)
            next_obs = torch.from_numpy(self.sequence[i + self.n_step][0]).cuda(self.actor_id%2+1).unsqueeze(0)
            next_action = self.target_actor(next_obs)

            q_value = self.critic(obs, action).detach().cpu().numpy()
            reward = self.sequence[i][2][0]
            terminal = self.sequence[i + self.n_step - 1][3][0]
            next_q_value = self.target_critic(next_obs, next_action).detach().cpu().numpy()
            
            if i >= self.burn_in_length:
                target_q_value = (reward + (self.gamma ** self.n_step) * (1.-terminal) * next_q_value)
                target_q_value = invertical_vf(torch.tensor(target_q_value).cuda(self.actor_id%2+1)).detach().cpu().numpy()
                self.td_loss.append((q_value - target_q_value).mean())
            if i >= self.sequence_length:
                self.priority.append(calc_priority(np.array(list(self.td_loss), dtype=np.float32) ** 2.))

    def run(self):
        episode = 0
        step = 0
        reward_sum = 0

        while True:
            time_step = self.env.reset()
            obs = get_obs(time_step.observation)
            self.actor.reset_state()
            self.critic.reset_state()
            self.target_actor.reset_state()
            self.target_critic.reset_state()
            self.sequence = []
            self.recurrent_state = []
            self.priority = []
            self.td_loss.clear()
            last_obs = None
            episode_step = 0
            done = False
            if self.actor_id == 0 and episode != 0:
                print('episode:', episode,'step:', step, 'reward:', reward_sum)
            episode += 1
            reward_sum = 0

            while not time_step.last():

                # get recurrent state
                actor_hx, actor_cx = self.actor.get_state()
                target_actor_hx, target_actor_cx = self.target_actor.get_state()
                critic_hx, critic_cx = self.critic.get_state()
                target_critic_hx, target_critic_cx = self.target_critic.get_state()
                
                action = self.actor(torch.from_numpy(obs).cuda(self.actor_id%2+1))
                target_action = self.target_actor(torch.from_numpy(obs).cuda(self.actor_id%2+1))
                _ = self.critic(torch.from_numpy(obs).cuda(self.actor_id%2+1), action)
                _ = self.target_critic(torch.from_numpy(obs).cuda(self.actor_id%2+1), target_action)

                action = action.detach().cpu().numpy()[0]
                action += np.random.normal(0, 0.3, (self.action_size))
                action = np.clip(action, -1, 1)

                reward = 0.
                sleep(0.01)
                for i in range(4):
                    time_step = self.env.step(action)
                    next_obs = get_obs(time_step.observation)
                    reward += time_step.reward
                    if time_step.last():
                        break

                reward_sum += reward
                step += 1
                episode_step += 1
                terminal = 1. if time_step.last() else 0.
                self.sequence.append((obs[0], action, [reward], [terminal]))
                obs = next_obs.copy()

                self.recurrent_state.append([[actor_hx[0], actor_cx[0]], [target_actor_hx[0], target_actor_cx[0]], 
                                                [critic_hx[0], critic_cx[0]], [target_critic_hx[0], target_critic_cx[0]]])

                if step % self.actor_parameter_update_interval == 0:
                    self.load_model()

            if len(self.sequence) >= self.sequence_length:
                self.sequence.extend([(np.zeros((self.obs_size), dtype=np.float32), np.zeros((self.action_size), dtype=np.float32), [0.], [1.]) for i in range(self.n_step)])
                self.calc_nstep_reward()
                self.calc_priorities()
                self.memory.add(self.sequence, self.recurrent_state, self.priority)
            
            if len(self.memory.memory) > self.memory_save_interval:
                self.memory.save(self.actor_id)

            