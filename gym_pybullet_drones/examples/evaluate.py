import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.EmpowermentAviary import EmpowermentAviary
from gym_pybullet_drones.envs.MultiEmpowermentAviary import MultiEmpowermentAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin_lidar') # 'kin' or 'rgb', kin observation space is Kinematic information (pose, linear and angular velocities) or 'kin_lidar' for kin + lidar
DEFAULT_ACT = ActionType('two_d_pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid' or 'two_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False


def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, checkpoint=None, checkpoint_folder=None, filename=None, eval_set=None):
    
    csvfilename = filename + '.csv'
    filename = os.path.join(output_folder, filename)       

    
    print("Filename: ", filename)

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)

    model = PPO.load(path)

    if eval_set is not None:
        print("Evaluating on ", eval_set)
        environment_set = pd.read_csv(eval_set)
        print(environment_set)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = EmpowermentAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video,
                               eval_set=environment_set)
        test_env_nogui = EmpowermentAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, eval_set=environment_set)
    else:
        test_env = MultiEmpowermentAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiEmpowermentAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )
    #print the shape of the observation space for debugging
    #print("Observation space shape: ", test_env.observation_space.shape)
    
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=1
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    
    eval_counter = 0
    obs, info = test_env.reset(seed=eval_counter, options={})
    
        
        
        
    #print obs for debugging
    start = time.time()
    with open(csvfilename, 'a') as f:
        f.write('time,x,y,z,vx,vy,vz\n')
    while True:
        #add a counter for time to the csv file
        i = time.time() - start
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN or DEFAULT_OBS == ObservationType.KINLID:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
                #save the x and y positions of the drone along with the timestamp to a csv file
                with open(csvfilename, 'a') as f:
                    f.write(str(i/test_env.CTRL_FREQ) + ',' + str(obs2[0]) + ',' + str(obs2[1]) + ',' + str(obs2[2]) + ',' + str(obs2[10]) + ',' + str(obs2[11]) + ',' + str(obs2[12]) + '\n')                
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            eval_counter += 1
            if eval_counter >= environment_set.shape[0]:
                break
            obs, info = test_env.reset(seed=eval_counter, options={})
        if truncated:
            eval_counter += 1
            if eval_counter >= environment_set.shape[0]:
                break
            obs, info = test_env.reset(seed=eval_counter, options={})        
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN or DEFAULT_OBS == ObservationType.KINLID:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    # add argument for the checkpoint folder path to load info
    parser.add_argument('--checkpoint_folder',  default=None,                  type=str,           help='Path to checkpoint folder for resuming training', metavar='')
    # add checkpoint argument for resuming training
    parser.add_argument('--checkpoint',         default=None,                  type=str,           help='Path to checkpoint file for resuming training from the checkpoint_folder', metavar='')
    # add argument for name of file to save the model
    parser.add_argument('--filename',           default=None,                  type=str,           help='Name of the file to evaluate the model', metavar='')
    # add argument for the name of the file of the evaluation set used
    parser.add_argument('--eval_set',           default=None,                  type=str,           help='Name of the file of the evaluation set used', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))