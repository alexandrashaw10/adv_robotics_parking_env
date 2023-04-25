from abc import abstractmethod
from gym import Env
from gym.envs.registration import register
import numpy as np
import random

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.parking_env import GoalEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env.utils import are_polygons_intersecting



class ParkingEnv2(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsWithGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -500,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 900,
            "screen_height": 1200,
            "centering_position": [0.5, 0.5],
            "scaling": 10,
            "controlled_vehicles": 1,
            "vehicles_count": 10,
            "add_walls": True, 
            "y_offset": 10,
            "length": 8,
            "spots": 14,
            "obstacles": False,
            "font_size": 22,
            "random_start": False,
            "exp_reward": True,
            'custom_reward': False,
            'custom_reward_scale': 1,
            "exp_position_weights": [0.04, 0.03, 0, 0, 0, 0],
            "exp_angle_weights": [0, 0, 0, 0, 40, 40],
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.config['observation']['vehicles_count'] = self.config['vehicles_count']

        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.config["observation"])

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv2, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        spots = self.config["spots"]
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = self.config["y_offset"]
        length = self.config["length"]

        second_row_offset = 10 if self.config['y_offset'] > 0 else 20
        second_row_y = 2 * y_offset + length + second_row_offset
        
        id = 0
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt, identifier=id, display_font_size=self.config['font_size'])) # Upper 1
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt, identifier=id+1, display_font_size=self.config['font_size'])) # Upper 2

            net.add_lane("a", "e", StraightLane([x, second_row_y], [x, second_row_y+length], width=width, line_types=lt, identifier=id+2, display_font_size=self.config['font_size'])) # Lower 1
            net.add_lane("c", "d", StraightLane([x, -second_row_y], [x, -second_row_y-length], width=width, line_types=lt, identifier=id+3, display_font_size=self.config['font_size'])) # Lower 2

            id += 4
        
        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])
        
        self.allowed_vehicle_space = {
            'x' : (-30, 30),
            'y' : [(y_offset+length+5, second_row_y-5), (-second_row_y+5, -y_offset-length-5)]
        }
        if self.config["y_offset"] >= 10:
            self.allowed_vehicle_space['y'].append((-self.config["y_offset"]//2, self.config["y_offset"]//2))

        # Walls
        x_end = abs((1 - spots // 2) * (width + x_offset) - width / 2)
        wall_y = second_row_y + length + 4 
        wall_x = x_end + 14

        for y in [-wall_y, wall_y]:
            obstacle = Obstacle(self.road, [0, y])
            obstacle.LENGTH, obstacle.WIDTH = (2*wall_x, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        for x in [-wall_x, wall_x]:
            obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
            obstacle.LENGTH, obstacle.WIDTH = (2*wall_y, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)

        if self.config["y_offset"] == 0: 
            # Draw a line in between center rows
            obstacle = Obstacle(self.road, [0, 0])
            obstacle.LENGTH, obstacle.WIDTH = (2*(x_end + width//2 + 1), 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            obstacle.collidable = False
            obstacle.line_color = (0, 83, 138)
            self.road.objects.append(obstacle)

        if self.config['obstacles']:
            self._create_obstacles()

    def _create_obstacles(self):
        """Create some random obstacles"""
        obstacle = Obstacle(self.road, (18, -18 - self.config['y_offset']), 90)
        obstacle.LENGTH, obstacle.WIDTH = 5, 5
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        obstacle.line_color = (187, 84, 49)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, (-18, 18 + self.config['y_offset']), 90)
        obstacle.LENGTH, obstacle.WIDTH = 5, 5
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        obstacle.line_color = (187, 84, 49)
        self.road.objects.append(obstacle)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            if self.config["random_start"]:
                while(True):
                    x = np.random.randint(self.allowed_vehicle_space['x'][0], self.allowed_vehicle_space['x'][1])
                    y_sector = np.random.choice([0, 1])
                    y = np.random.randint(self.allowed_vehicle_space['y'][y_sector][0], self.allowed_vehicle_space['y'][y_sector][1])
                    vehicle = self.action_type.vehicle_class(self.road, [x, y], 2*np.pi*self.np_random.uniform(), 0)

                    intersect = False
                    for o in self.road.objects:
                        res, _, _ = are_polygons_intersecting(vehicle.polygon(), o.polygon(), vehicle.velocity, o.velocity)
                        intersect |= res
                    if not intersect:
                        break
            else:
                vehicle = self.action_type.vehicle_class(self.road, [0, 0], 2*np.pi*self.np_random.uniform(), 0)

            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Goal
        goal_lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, goal_lane.position(goal_lane.length/2, 0), heading=goal_lane.heading)
        self.road.objects.append(self.goal)

        # Other vehicles
        free_lanes = self.road.network.lanes_list().copy()
        free_lanes.remove(goal_lane)
        random.Random(4).shuffle(free_lanes)
        for _ in range(self.config["vehicles_count"] - 1):
            lane = free_lanes.pop()
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)

    def _custom_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        rel_achieved = achieved_goal * self.config["observation"]["scales"]
        des_goal = np.dot(desired_goal, self.config["observation"]["scales"])
        reward = 2 * np.exp(-0.05 * (np.dot(np.square(rel_achieved - des_goal), self.config["reward_weights"])))
        return reward

    def _custom_reward_mathworks(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        # multiply each position by the scales so that it matches the scale we want
        scal_achieved = achieved_goal * self.config["observation"]["scales"]
        scal_goal = desired_goal * self.config["observation"]["scales"]

        diff = scal_achieved - scal_goal
        
        pos_rew = np.dot(np.square(diff), self.config["exp_position_weights"])
        angle_rew = np.dot(np.abs(diff), self.config["exp_angle_weights"])

        r = 8 * np.exp(-pos_rew) + 0.5 * np.exp(-angle_rew) - self.config["collision_reward"] / self.config["duration"]
        return r
  
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        if self.config['custom_reward']:
            return self._custom_reward(achieved_goal, desired_goal)
        elif self.config['exp_reward']:
            return self._custom_reward_mathworks(achieved_goal, desired_goal)
        else:
            return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        timeout = self.time >= self.config["duration"]
        return bool(crashed or success or timeout)


# class ParkingEnvActionRepeat(ParkingEnv):
#     def __init__(self):
#         super().__init__({"policy_frequency": 1, "duration": 20})

# class ParkingEnvParkedVehicles(ParkingEnv):
#     def __init__(self):
#         super().__init__({"vehicles_count": 10})

# class ParkingEnvSmallerLot(ParkingEnv):
#     def __init__(self):
#         super().__init__({"y_offset": 5})
        
# class ParkingEnvSmallerLotWV(ParkingEnv):
#     def __init__(self):
#         super().__init__({"y_offset": 5, "vehicles_count": 4})
       


register(
    id='parking-v2',
    entry_point='highway_env.envs:ParkingEnv2',
)

# register(
#     id='parking-ActionRepeat-v0',
#     entry_point='highway_env.envs:ParkingEnvActionRepeat'
# )

# register(
#     id='parking-parkedVehicles-v0',
#     entry_point='highway_env.envs:ParkingEnvParkedVehicles'
# )

# register(
#     id='parking-smallerLot-v0',
#     entry_point='highway_env.envs:ParkingEnvSmallerLot'
# )

# register(
#     id='parking-smallerLot-v1',
#     entry_point='highway_env.envs:ParkingEnvSmallerLotWV'
# )
