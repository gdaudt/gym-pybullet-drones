import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class EmpowermentAviary(BaseRLAviary):
    """ Single or multi-agent RL problem: reaching a target position while maximizing empowerment. """
    ####################################################################
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.PID
                 ):
        """Initialization of a single or multi-agent RL environment.

        Using the generic single or multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The number of drones.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([1,1,1])
        self.EPISODE_LEN_SEC = 12
        self.OBSTACLES = []
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
        
####################################################################

    #add obstacles to the environment
    def _addObstacles(self):
        obstacle_id = p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -1, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        
####################################################################

    # override the reset method to spawn obstacles
    def reset(self, seed = None, options = None):
        initial_obs, initial_info = super().reset(seed, options)
        self._addObstacles()
        return initial_obs, initial_info
        
####################################################################

    # compute the reward for the agent, which is the euclidean distance to the target position
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = np.linalg.norm(state[0:3]-self.TARGET_POS, axis=-1)
        reward = np.exp(-ret)
        # print("State: ", state[0:3])
        # print("Target: ", self.TARGET_POS)
        # print("Distance: ", ret)
        # print("Reward: ", reward)
        return reward

####################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            print("Terminated because drone reached target")
            return True
        else:
            return False
        # return False

####################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            print("Truncated because drone is too far away or tilted")
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("Truncated because episode length exceeded")
            return True
        else:
            return False

####################################################################

    def _computeInfo(self):
        """Computes the current info value.

        Returns
        -------
        dict
            The info dictionary.

        """
        return {"answer": 42} # The answer to the ultimate question of life, the universe, and everything