try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)


class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad(): 
                return int(torch.argmax(self.forward(state)))


# class DQN(BaseAgent):
#     def construct(self):
#         self.layers = nn.Sequential(
#             nn.Linear(self.feature_size(), 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.num_actions)
#         )

# class ConvDQN(DQN):
#     def construct(self):
#         self.features = nn.Sequential(
#             nn.Conv2d(self.input_shape[0], 16, kernel_size=3,stride=(1,2)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=(1,2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=2, stride=(1,2)),
#             nn.ReLU()
#         )
#         super().construct()

class DQN(BaseAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=(1,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        super().construct()


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


    
class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        test_case_id = kwargs.get('test_case_id')

    def initialize(self, **kwargs):
        env                 = kwargs.get('env') # WARNING: not available in the 2nd task
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        self.model = get_model()

    def step(self, state, *args, **kwargs):
        return self.model.act(state)

    def update(self, *args, **kwargs):
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')

def create_agent(test_case_id, *args, **kwargs):
    return DQNAgent()



def test(agent, env, runs=1000, t_max=100):
    rewards = []
    for run in range(runs):
        state = env.reset()
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
        agent.initialize(**agent_init)
        episode_rewards = 0.0

        for t in range(t_max):
            action = agent.step(state)

            next_state, reward, done, info = env.step(action)
            full_state = {
                'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                'done': done, 'info': info
            }
            agent.update(**full_state)
            state = next_state
            
            episode_rewards += reward
            
            if done:    
                break
        rewards.append(episode_rewards)

    avg_rewards = sum(rewards)/len(rewards)

    print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
    
    return avg_rewards

def timed_test(task):
    start_time = time.time()
    rewards = []

    for tc in task['testcases']:
        agent = create_agent(tc['id']) # `test_case_id` is unique between the two task 
        print("[{}]".format(tc['id']), end=' ')
        avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
        rewards.append(avg_rewards)
    point = sum(rewards)/len(rewards)
    elapsed_time = time.time() - start_time

    print('Point:', point)

    for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
        if elapsed_time < task['time_limit'] * t:
            print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
            print("WARNING: do note that this might not reflect the runtime on the server.")
            break

def get_task(task_id):
    if task_id == 1:
        test_case_id = 'task1_test'
        return { 
            'time_limit' : 600,
            'testcases' : [{'id' : test_case_id, 'env' : construct_task1_env(), 'runs' : 1, 't_max' : 50}]
            }
    elif task_id == 2:
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }
    else:
        raise NotImplementedError

if __name__ == '__main__':
    import sys
    import time
    from env import construct_task1_env, construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    try:
        task_id = int(sys.argv[1])
    except:
        print('Run agent on an example task.')
        print('Usage: python __init__.py <task number>')
        print('Example:\n   python __init__.py 2')
        exit()

    print('Testing on Task {}'.format(task_id))

    task = get_task(int(task_id))
    timed_test(task)




