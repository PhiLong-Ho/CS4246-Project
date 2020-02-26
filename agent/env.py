import gym
import numpy as np
import random
from gym_grid_driving.envs.grid_driving import LaneSpec

def construct_task2_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -2]), 
                        LaneSpec(cars=7, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)


def construct_train_task2_env():
    random_car = numpy.random.randint(5, 10, 10)
    speed_range = [[-3,-3], [-3, -2], [-3, -1], [-2, -2], [-2, -1], [-1, -1]]
    random_speed_range = random.sample(speed_range, 10)

    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=random_car[0], speed_range=random_speed_range[0]), 
                        LaneSpec(cars=random_car[1], speed_range=random_speed_range[1]), 
                        LaneSpec(cars=random_car[2], speed_range=random_speed_range[2]), 
                        LaneSpec(cars=random_car[3], speed_range=random_speed_range[3]), 
                        LaneSpec(cars=random_car[4], speed_range=random_speed_range[4]), 
                        LaneSpec(cars=random_car[5], speed_range=random_speed_range[5]), 
                        LaneSpec(cars=random_car[6], speed_range=random_speed_range[6]), 
                        LaneSpec(cars=random_car[7], speed_range=random_speed_range[7]), 
                        LaneSpec(cars=random_car[8], speed_range=random_speed_range[8]), 
                        LaneSpec(cars=random_car[9], speed_range=random_speed_range[9])]
              }           
    return gym.make('GridDriving-v0', **config)

def construct_task1_env() :
    test_config = [{'lanes' : [LaneSpec(10, [-2, -2])] *2 + [LaneSpec(10, [-3, -3])] *2 +
                              [LaneSpec(8, [-4, -4])] *2 + [LaneSpec(8, [-5, -5])] *2 +
                              [LaneSpec(12, [-4, -4])] *2 + [LaneSpec(12, [-3, -3])] *2 ,
                   'width' :50,
                   'agent_speed_range' : (-3,-1),
                   'random_seed' : 11}]
    test_index = 0
    case = test_config[test_index]
    return gym.make('GridDriving-v0', **case)
