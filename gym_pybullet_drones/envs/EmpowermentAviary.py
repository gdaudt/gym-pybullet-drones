import numpy as np
import pybullet as p
import pybullet_utils


from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import ConvexHull

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
                 obs: ObservationType=ObservationType.KINLID,
                 act: ActionType=ActionType.PID,
                 numrays: int = 36,
                 lidar_angle: float = 2*np.pi,
                 max_range: float = 3.0
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
        self.TARGET_POS = np.array([3, 0, 1])
        self.EPISODE_LEN_SEC = 40
        self.OBSTACLES = []
        self.LIDAR_DATA = []
        self.OBSERVATION_TYPE = obs
        self.rayMissColor = [0, 1, 0]
        self.rayHitColor = [1, 0, 0]
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.rayLen = 3
        self.numRays = 4
        self.replaceLines = False   
        self.createObstacleLookup = True
        self.obstacleLookup = {}
        self.ndeg = np.pi/2
        
        self.LIDAR_NUM_RAYS = numrays
        self.LIDAR_ANGLE = lidar_angle
        self.LIDAR_MAX_RANGE = max_range
        
        # mode of trajectory sampling
        # 1 = chebyshev integrator, 2 = fourier series
        self.SAMPLING = 2
        # constants for trajectory sampling
        self.MASS = 0.027 # from the CF2X model urdf file
        # gravity force added to the maximum thrust force, taken from the CF2X model urdf file
        self.F_MAX = 0.027 * 9.81 + 0.027 * 8.33 # from max speed being 30 km/h over 1s
        # number of chebychev basisfunctions or fourier series terms
        self.N = 10
        # end time of the trajectory
        self.T_END = 1
        # omega for fourier series
        self.omega = 2*np.pi / self.T_END
        # number of points in the trajectory
        self.N_POINTS = 100
        #number of trajectories sampled
        self.N_TRAJECTORIES = 50
        self.T_SPACED = np.linspace(0, self.T_END, self.N_POINTS)
        
        
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
        
       
        # obstacle1pos = [-1, -1, .5]
        # obstacle2pos = [.5, 1, .5]
        # obstacle3pos = [.5, 2, .5]
        # obstacle4pos = [-1, 2, .5]
        #[up+-down, left+-right, z]
        z = 1.05
        o = 0
        obstacles= ([[1, 1, z], [2, 1, z], [1, 0, z], [2, 0, z], [1, -2, z], [2, -2, z]])
        xoffset = 0.5
        yoffset = 0.25
        zoffset = 1
        #spawn a small object at the target position
        # target_id = p.loadURDF("cube_small.urdf",                               
        #                        self.TARGET_POS,
        #                         p.getQuaternionFromEuler([0, 0, 0]),
        #                        )
        for obstaclePosition in obstacles:
            obstacle_id = p.loadURDF("cube_small.urdf",
                       obstaclePosition[0:3],
                       p.getQuaternionFromEuler([0, 0, o]),
                       physicsClientId=self.CLIENT
                       )
                      
            if(self.createObstacleLookup):
                # for each obstacle, store the min and max corners for AABB collision detection, indexed by the object id
                # in each obstacle, min corner is coordinates x+xoffset, y+yoffset, z-zoffset, and max corner is x-xoffset, y-yoffset, z+zoffset
                # add the min an max corners to the obstacleLookup dictionary
                self.obstacleLookup[obstacle_id] = [[obstaclePosition[0]+xoffset, obstaclePosition[1]+yoffset, obstaclePosition[2]-zoffset], [obstaclePosition[0]-xoffset, obstaclePosition[1]-yoffset, obstaclePosition[2]+zoffset], obstaclePosition]
                print("Obstacle: ", obstacle_id, " positions: ", self.obstacleLookup[obstacle_id])            
        self.createObstacleLookup = False
####################################################################

    # override the reset method to spawn obstacles
    # def reset(self, seed = None, options = None):
    #     initial_obs, initial_info = super().reset(seed, options)        
    #     self.reset_lidar()
    #     return initial_obs, initial_info

####################################################################

    def reset_lidar(self):
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.replaceLines = False

####################################################################

    #simulate a lidar sensor casting rays in a circular pattern
    def simulateLidar(self, max_distance=3.0):
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        #generate start and end points for the rays
        start_points = np.tile(drone_pos, (self.numRays, 1))
        end_points = []
        
        for angle in np.linspace(0, 2 * np.pi, self.numRays, endpoint=False):
            x = drone_pos[0] + max_distance * np.cos(angle)
            y = drone_pos[1] + max_distance * np.sin(angle)
            z = drone_pos[2]  # Keep the z-height constant
            end_points.append([x, y, z])
            
        #perform the ray test
        results = p.rayTestBatch(start_points, end_points, physicsClientId=self.CLIENT)
        #print("LIDAR DATA: ", results)
        #store the results
        self.LIDAR_DATA = results
        #visualize the Lidar
        if(self.GUI):
            self.visualizeLidar(results, start_points, end_points)
        
####################################################################

    #visualize the lidar data
    def visualizeLidar(self, results, start_points, end_points):
        for i in range (self.numRays):
            hitObjectUid = results[i][0]
            hitFraction = results[i][2]
            hitPosition = results[i][3]
            if (hitFraction==1.):
                if(self.replaceLines == False):
                    self.rayIds.append(p.addUserDebugLine(start_points[i], end_points[i], self.rayMissColor, lineWidth=1, lifeTime=0.1))
                    print("RayId: ", self.rayIds)
                else:
                    p.addUserDebugLine(start_points[i], end_points[i], self.rayMissColor, lineWidth=1, lifeTime=0.1, replaceItemUniqueId=self.rayIds[i])
            else:
                localHitTo = [start_points[i][0] + hitFraction * (end_points[i][0] - start_points[i][0]),
                              start_points[i][1] + hitFraction * (end_points[i][1] - start_points[i][1]),
                              start_points[i][2] + hitFraction * (end_points[i][2] - start_points[i][2])]
                if(self.replaceLines == False):
                    self.rayIds.append(p.addUserDebugLine(start_points[i], localHitTo, self.rayHitColor, lineWidth=1, lifeTime=0.1))
                else:
                    p.addUserDebugLine(start_points[i], localHitTo, self.rayHitColor, lineWidth=1, lifeTime=0.1, replaceItemUniqueId=self.rayIds[i])
        self.replaceLines = True
                
####################################################################

    #extend the step function to simulate the lidar sensor
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # if self.OBSERVATION_TYPE == ObservationType.KINLID:
        #     #print only lidar information from observation
        #     print("LIDAR DATA: ", obs[0][57:])
        # save simulate lider for later
        #self.simulateLidar()
        return obs, reward, terminated, truncated, info

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
        empowerment = self._computeEmpowerment(state)
        # print("current velocity: ", state[10:13])
        # print("current position: ", state[0:3])
        # print("current empowerment: ", empowerment)
        return (reward*1.5) * (empowerment)
        #return reward

####################################################################

    # compute a trajectory based on current state and chebyshev polynomials
    # returns an array with the x, y, z positions of the final point of the trajectory
    def _computeTrajectory(self, state):
       
       
        if self.SAMPLING == 1:
            F_x = self.F_MAX/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=self.N)-1)/(self.N/3), domain=[0, self.T_END], window=[-1, 1])
            F_y = self.F_MAX/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=self.N)-1)/(self.N/3), domain=[0, self.T_END], window=[-1, 1])
            F_z = self.F_MAX/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=self.N)-1)/(self.N/3), domain=[0, self.T_END], window=[-1, 1])
            #compute acceleration
            a_x = F_x(self.T_SPACED)/self.MASS
            a_y = F_y(self.T_SPACED)/self.MASS
            a_z = F_z(self.T_SPACED)/self.MASS
            # integrate acceleration to get velocity, adding it to the current velocity
            # cur_pos=state[0:3],
            # cur_quat=state[3:7],
            # cur_vel=state[10:13],
            # cur_ang_vel=state[13:16]
            v_x = state[10] + cumulative_trapezoid(a_x, self.T_SPACED, initial=0)
            v_y = state[11] + cumulative_trapezoid(a_y, self.T_SPACED, initial=0)
            v_z = state[12] + cumulative_trapezoid(a_z, self.T_SPACED, initial=0)
            # integrate velocity to get position, adding it to the current position
            x = state[0] + cumulative_trapezoid(v_x, self.T_SPACED, initial=0)
            y = state[1] + cumulative_trapezoid(v_y, self.T_SPACED, initial=0)
            z = state[2] + cumulative_trapezoid(v_z, self.T_SPACED, initial=0)
            
            #print("Final position: ", [x[-1], y[-1], z[-1]])
            return np.array([x[-1], y[-1], z[-1]])
        
        elif self.SAMPLING == 2:
            # Random Fourier coefficients within force limits
            A_x = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            B_x = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            A_y = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            B_y = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            A_z = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            B_z = np.random.uniform(-self.F_MAX, self.F_MAX, self.N)
            
            # Compute acceleration as a Fourier series
            a_x = np.sum([A_x[n] * np.cos((n+1) * self.omega * self.T_SPACED) + B_x[n] * np.sin((n+1) * self.omega * self.T_SPACED) for n in range(self.N)], axis=0) / self.MASS
            a_y = np.sum([A_y[n] * np.cos((n+1) * self.omega * self.T_SPACED) + B_y[n] * np.sin((n+1) * self.omega * self.T_SPACED) for n in range(self.N)], axis=0) / self.MASS
            a_z = np.sum([A_z[n] * np.cos((n+1) * self.omega * self.T_SPACED) + B_z[n] * np.sin((n+1) * self.omega * self.T_SPACED) for n in range(self.N)], axis=0) / self.MASS
            
            # Integrate acceleration to get velocity
            v_x = state[10] + np.cumsum(a_x) * (self.T_END / self.N_POINTS)
            v_y = state[11] + np.cumsum(a_y) * (self.T_END / self.N_POINTS)
            v_z = state[12] + np.cumsum(a_z) * (self.T_END / self.N_POINTS)

            # Integrate velocity to get position
            x = state[0] + np.cumsum(v_x) * (self.T_END / self.N_POINTS)
            y = state[1] + np.cumsum(v_y) * (self.T_END / self.N_POINTS)
            z = state[2] + np.cumsum(v_z) * (self.T_END / self.N_POINTS)
            
            return np.array([x[-1], y[-1], z[-1]])
        
        else:
            print("Wrong sampling mode chosen")
            
        return np.array([0, 0, 0])

####################################################################

    def _computeCollision(self, final_pos):
        """Computes the current collision value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if(final_pos[2] >= 2 or final_pos[2] <= 0):
                #print("Collision with ground or ceiling")
                return True
        
        #check if the final position is inside an obstacle
        for obstacle_id in self.obstacleLookup:
            #if distance between object is too big, continue
            if np.linalg.norm(self.obstacleLookup[obstacle_id][2]-final_pos) > 2:
                continue
            min_corner = self.obstacleLookup[obstacle_id][0]
            max_corner = self.obstacleLookup[obstacle_id][1]
            # if the final position is inside the obstacle or on ground (z<=0), return True
            # print all corners and the final position          
            # add artificial collision to ceiling at z = 2
            
            if (min_corner[0] <= final_pos[0] <= max_corner[0] and
                min_corner[1] <= final_pos[1] <= max_corner[1] and
                min_corner[2] <= final_pos[2] <= max_corner[2]):
                
                #print("Collision with obstacle")
                return True
        return False

####################################################################


    #compute the empowerment of the agent
    def _computeEmpowerment(self, state):
        
        final_points = None
        
        for _ in range(self.N_TRAJECTORIES):
            computed_trajectory = self._computeTrajectory(state)
            if(self._computeCollision(computed_trajectory)):
                continue
            else:
                if final_points is None:
                    final_points = np.array([computed_trajectory])
                else:
                    final_points = np.vstack((final_points, computed_trajectory))
        
        # if final_points have enough points to compute a convex hull (at least 4), compute the empowerment
        if final_points is not None and final_points.shape[0] > 3:
            hull = ConvexHull(final_points)
            empowerment = np.log(hull.volume)
            #print how many trajectories did not collide
            # print("Number of trajectories safe: ", final_points.shape[0])
            return empowerment        
        return 0.0
     
####################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """    

        state = self._getDroneStateVector(0)       
        
        p.performCollisionDetection()
        #check if the final position is inside an obstacle
        for obstacle_id in self.obstacleLookup:
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=obstacle_id, physicsClientId=self.CLIENT)
            #check if contact points are not empty and if the contact force is more than 0. Only terminate if drone collides when non-stationary to avoid false positives
            if len(contact_points) > 0:
                for contact in contact_points:
                    if contact[9] > 0:                       
                        print("Terminated due to collision with obstacle ", obstacle_id)
                        return True        
        
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
        if (abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] > 3.0 # Truncate when the drone is too far away
             or abs(state[7]) > .9 or abs(state[8]) > .9 # Truncate when the drone is too tilted
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