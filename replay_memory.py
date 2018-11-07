import torch
import torch.utils.data
from collections import deque
import os
from time import time, sleep
import numpy as np

class ReplayMemory:
    def __init__(self, memory_sequence_size = 100000, batch_size = 64):
        self.path = './memory_data/'
        self.memory_sequence_size = memory_sequence_size
        self.batch_size = batch_size
        self.memory = deque()
        self.priority = deque()
        self.total_priority = deque()
        self.recurrent_state = deque()

        self.burn_in_length = 20 # 40-80
        self.learning_length = 40
        self.n_step = 5

    def add(self, episode, recurrent_state, priority):
        self.memory.append(episode)
        self.recurrent_state.append(recurrent_state)
        self.priority.append(priority)
        self.total_priority.append(sum(list(priority)))
    
    def clear(self):
        self.memory.clear()
        self.recurrent_state.clear()
        self.priority.clear()
        self.total_priority.clear()
    
    def size(self):
        return sum([len(self.memory[i]) for i in range(len(self.memory))])

    def save(self, actorID):
        if os.path.isfile(self.path + 'memory.pt'):
            try:
                memory = torch.load(self.path + 'memory{}.pt'.format(actorID))
                memory['replay_memory'].extend(self.memory)
                memory['recurrent_state'].extend(self.recurrent_state)
                memory['priority'].extend(self.priority)
                memory['total_priority'].extend(list(self.total_priority))
                torch.save(memory, self.path + 'memory{}.pt'.format(actorID))
            except:
                sleep(10)
                memory = torch.load(self.path + 'memory{}.pt'.format(actorID))
                memory['replay_memory'].extend(self.memory)
                memory['recurrent_state'].extend(self.recurrent_state)
                memory['priority'].extend(self.priority)
                memory['total_priority'].extend(list(self.total_priority))
                torch.save(memory, self.path + 'memory{}.pt'.format(actorID))
        else:
            memory = {'replay_memory': self.memory,
                      'recurrent_state': self.recurrent_state,
                      'priority': self.priority,
                      'total_priority': list(self.total_priority)}
            torch.save(memory, self.path + 'memory{}.pt'.format(actorID))

        self.memory.clear()
        self.recurrent_state.clear()
        self.priority.clear()
        self.total_priority.clear()


class LearnerReplayMemory:
    def __init__(self, memory_sequence_size = 500000, batch_size = 32):
        self.path = './memory_data/'
        self.memory_sequence_size = memory_sequence_size
        self.sequence_counter = 0
        self.batch_size = batch_size
        self.memory = deque()
        self.recurrent_state = deque()
        self.priority = deque()
        self.total_priority = deque()

        self.burn_in_length = 20 # 40-80
        self.learning_length = 40
        self.sequence_length = self.burn_in_length + self.learning_length
        self.n_step = 5
    
    def size(self):
        return sum([len(self.memory[i]) for i in range(len(self.memory))])
    
    def get(self, index):
        return self.memory[index]
    
    def clear(self):
        self.memory.clear()
        self.recurrent_state.clear()
        self.priority.clear()
        self.total_priority.clear()

    def get_weighted_sample_index(self):
        total_priority = torch.tensor(np.array(self.total_priority), dtype=torch.float)
        return torch.utils.data.WeightedRandomSampler(total_priority, self.batch_size, replacement=True)
    
    def sample(self):
        # エピソードのインデックスを取得
        sample_episode_index = self.get_weighted_sample_index()
        sample_episode_index = [index for index in sample_episode_index]

        # 各エピソードの中からサンプルするシーケンスのインデックスを取得
        # batch * sequence * elements(obs, action, reward, done)
        sample_sequence_index = []
        trajectory_sequence_batch = []
        rnn_state_batch = []
        for episode_index in sample_episode_index:
            episode_trajectory = self.memory[episode_index]
            priority = torch.tensor(np.array(self.priority[episode_index]), dtype=torch.float)
            sequence_index = torch.utils.data.WeightedRandomSampler(priority, 1, replacement = True)
            sequence_index = [index for index in sequence_index]
            sequence_index = sequence_index[0]
            sample_sequence_index.append(sequence_index)
            trajectory_sequence_batch.append(episode_trajectory[sequence_index: sequence_index + self.sequence_length+self.n_step])

            episode_rnn_state = self.recurrent_state[episode_index]
            rnn_state_batch.append(episode_rnn_state[sequence_index])

        # elements(obs, action, reward, terminal) * sequence * batch
        trajectory_batch_sequence = [[[trajectory_sequence_batch[b][s][e] for b in range(self.batch_size)] for s in range(self.sequence_length+self.n_step)] for e in range(4)]
        obs_batch_sequence = torch.tensor(trajectory_batch_sequence[0]).cuda()
        action_batch_sequence = torch.tensor(trajectory_batch_sequence[1]).cuda()
        reward_batch_sequence = torch.tensor(trajectory_batch_sequence[2]).cuda()
        terminal_batch_sequence = torch.FloatTensor(trajectory_batch_sequence[3]).cuda()

        # batch * state -> state * batch
        rnn_state_batch = [[[rnn_state_batch[b][e][i] for b in range(self.batch_size)] for i in range(2)] for e in range(4)]
        actor_state_batch = torch.tensor(rnn_state_batch[0]).cuda()
        target_actor_state_batch = torch.tensor(rnn_state_batch[1]).cuda()
        critic_state_batch = torch.tensor(rnn_state_batch[2]).cuda()
        target_critic_state_batch = torch.tensor(rnn_state_batch[3]).cuda()

        return sample_episode_index, sample_sequence_index, obs_batch_sequence, action_batch_sequence, reward_batch_sequence, terminal_batch_sequence, \
                actor_state_batch, target_actor_state_batch, critic_state_batch, target_critic_state_batch

    def load(self, actorID):
        if os.path.isfile(self.path + 'memory{}.pt'.format(actorID)):
            try:
                memory_dict = torch.load(self.path + 'memory{}.pt'.format(actorID))
                self.memory.extend(memory_dict['replay_memory'])
                self.recurrent_state.extend(memory_dict['recurrent_state'])
                self.priority.extend(memory_dict['priority'])
                self.total_priority.extend(memory_dict['total_priority'])
                # シーケンスの数は、エピソードの長さから重複する部分を取り除いたもの
                self.sequence_counter += sum([len(episode) - (self.sequence_length+self.n_step-1) for episode in memory_dict['replay_memory']])
                while self.sequence_counter > self.memory_sequence_size:
                    self.sequence_counter -= len(self.memory.popleft()) - (self.sequence_length)
                    self.recurrent_state.popleft()
                    self.priority.popleft()
                    self.total_priority.popleft()
                memory_dict['replay_memory'].clear()
                memory_dict['recurrent_state'].clear()
                memory_dict['priority'].clear()
                memory_dict['total_priority'].clear()
                torch.save(memory_dict, self.path + 'memory{}.pt'.format(actorID))
            except:
                sleep(np.random.rand() * 5 + 2)
                memory_dict = torch.load(self.path + 'memory{}.pt'.format(actorID))
                self.memory.extend(memory_dict['replay_memory'])
                self.recurrent_state.extend(memory_dict['recurrent_state'])
                self.priority.extend(memory_dict['priority'])
                self.total_priority.extend(memory_dict['total_priority'])
                self.sequence_counter += sum([len(episode) - (self.sequence_length+self.n_step-1) for episode in memory_dict['replay_memory']])
                while self.sequence_counter > self.memory_sequence_size:
                    self.sequence_counter -= len(self.memory.popleft()) - (self.sequence_length)
                    self.recurrent_state.popleft()
                    self.priority.popleft()
                    self.total_priority.popleft()
                memory_dict['replay_memory'].clear()
                memory_dict['recurrent_state'].clear()
                memory_dict['priority'].clear()
                memory_dict['total_priority'].clear()
                torch.save(memory_dict, self.path + 'memory{}.pt'.format(actorID))