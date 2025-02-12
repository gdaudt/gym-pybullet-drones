"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes EmpowermentAviary and MultiEmpowermentAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
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

DEFAULT_OBS = ObservationType('kin_lidar') # 'kin' or 'rgb' or 'kin_lidar', kin observation space is Kinematic information (pose, linear and angular velocities) and kin_lidar is Kinematic information and LIDAR data
DEFAULT_ACT = ActionType('two_d_pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid' or 'two_d_pid'
DEFAULT_AGENTS = 2  
DEFAULT_MA = False

class CheckpointCallback(BaseCallback):
    #### Callback for saving a model every n steps #############
    def __init__(self, save_freq, save_path, verbose = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}.zip")
            self_model = self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {checkpoint_file}")
        return True

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, checkpoint=None, checkpoint_folder=None, filename=None):


    if not multiagent:
        train_env = make_vec_env(EmpowermentAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = EmpowermentAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiEmpowermentAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiEmpowermentAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### if not resuming training from a checkpoint, create a new folder
    if checkpoint and checkpoint_folder:
        filename = checkpoint_folder
    elif filename:
        # save the model as save-+datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + filename
        filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+'-'+filename)       
    else:
        filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    print("Filename: ", filename)
        
    if checkpoint and os.path.isfile(os.path.join(filename, checkpoint)):
        print(f"[INFO] Resuming training from checkpoint {checkpoint}")
        model = PPO.load(checkpoint, train_env, verbose=1)
        print("testing if this goes beyond load")
    else:
    #### Train the model #######################################
        model = PPO('MlpPolicy',
                    train_env,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)

    checkpoint_path = os.path.join(filename, "checkpoints")
    
    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 10000. if not multiagent else 920.
        
    #add checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=checkpoint_path, verbose=1)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    model.learn(total_timesteps=int(1e6) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=callback_list,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = EmpowermentAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = EmpowermentAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
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

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=2
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
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
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
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
    parser.add_argument('--filename',           default=None,                  type=str,           help='Name of the file to save the model', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
