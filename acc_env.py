import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from vehicle_model import VehicleParam, EgoVehicle, ObjVehicle
from lane_model import Lane

class AccParam():
    def __init__(self):
        self.time_interval = 2.0 # 跟车时距
        self.safe_distance = 3.5 # 安全距离
        self.t_react = 0.5 # 人类驾驶员反应时间0.5s
        self.at_min = -2.0 # 制动下限
        self.at_max = 1.5  # 提速上限
        self.jerk_min = -2.5
        self.jerk_max = 2.0
        self.cushion_dis = 0.0 # 制动后的缓冲距离

class AccEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        super(AccEnv, self).__init__()
        action_acc_range = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]) # 动作空间: 加速度(离散)
        state_low  = np.array([0.0,   0.0,   -3.0, 0.0,   -3.0]) # 观察空间状态值下限, [ds_min, v_veh_min, a_veh_min, v_obj_min, a_obj_min]
        state_high = np.array([200.0, 120.0, 2.0,  120.0, 2.0])  # 观察空间状态值上限, [ds_max, v_veh_max, a_veh_max, v_obj_max, a_obj_max]
        self.action_space = spaces.Discrete(action_acc_range.shape[0])                        # 动作空间(离散)
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32) # 状态空间(连续)

        self.state = np.zeros(state_low.shape[0], dtype=np.float32)
        self.done = False

        # acc参数
        self.acc_param = AccParam()

        # 环境模型
        self.vehicle_param = VehicleParam(4.935, 1.915, 1.495, 2.915, 1.042, 1.191) # 车辆参数
        self.ego_vehicle = EgoVehicle(self.vehicle_param) # 主车
        self.obj_vehicle = ObjVehicle(self.vehicle_param) # 目标
        self.lane = Lane(width=10.0, length=10000.0)      # 车道
        self.ego_vehicle.set_state(0.0,   0.0, 0.0, 0.0)  # 主车初始状态
        self.obj_vehicle.set_state(20.0, 0.0, 0.0, 10.0) # 目标初始状态

        # 可视化
        self.fig = plt.figure('ACC Simulaion')                    # 画布
        self.vis_ax = self.fig.add_axes([0.03, 0.03, 0.93, 0.93]) # 绘图坐标轴

    def reset(self):
        self.state = np.zeros(self.state.shape[0], dtype=np.float32)
        self.done  = False

    def step(self, action):
        # TODO(wangjintao): 更新小车状态
        ego_delta = 0.0
        ego_acc = action
        self.ego_vehicle.step(ego_delta, ego_acc)
        self.obj_vehicle.step(0.0, 0.0)

        # TODO(wangjintao): 判断是否结束(失败或成功都结束)
        # TODO(wangjintao): 获取reward
        reward = 0.0
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        plt.axis('equal')
        self.lane.draw(self.vis_ax, [np.min([self.ego_vehicle.x, self.obj_vehicle.x]) - 20.0, np.max([self.ego_vehicle.x, self.obj_vehicle.x]) + 50.0])
        self.ego_vehicle.draw(self.vis_ax, 'tab:blue')
        self.obj_vehicle.draw(self.vis_ax, 'tab:green')
        self.fig.canvas.draw_idle()    # 重绘图形
        self.fig.canvas.flush_events() # 刷新

    # TODO(wangjintao): 验证奖励函数
    def reward_funcion(self):
        w_follow  = 0.3
        w_safety  = 0.5
        w_comfort = 0.2

        # 跟随性奖励
        reward_follow = -1.0
        reward_follow_distance = -1.0
        reward_follow_velocity = -1.0

        actual_follow_distance = self.obj_vehicle.x - self.ego_vehicle.x
        target_follow_distance = self.acc_param.time_interval * self.ego_vehicle.v + self.acc_param.safe_distance
        reward_follow_distance = -np.tanh(np.fabs(actual_follow_distance - target_follow_distance))

        reward_follow_velocity = -np.tanh(np.fabs(self.ego_vehicle.v - self.obj_vehicle.v))
        reward_follow = np.min([reward_follow_distance, reward_follow_velocity])

        # 安全性奖励
        reward_safety = -5.0
        d_star = self.ego_vehicle.v * self.acc_param.t_react + self.ego_vehicle.v * self.ego_vehicle.v / (2 * self.acc_param.at_min) +\
                 self.acc_param.cushion_dis - self.obj_vehicle.v * self.obj_vehicle.v / (2 * self.acc_param.at_min)
        reward_safety = -5.0 / 1 + np.exp((actual_follow_distance - d_star) / 2.0)

        # 舒适性奖励
        reward_comfort = -1.0
        reward_comfort_at = -1.0
        reward_comfort_jerk = -1.0
        if self.ego_vehicle.at >= 0.0:
            reward_comfort_at = -2 / (np.exp(self.acc_param.at_max - self.ego_vehicle.at) * np.exp(self.acc_param.at_max - self.ego_vehicle.at))
        else:
            reward_comfort_at = -2 / (np.exp(self.ego_vehicle.at - self.acc_param.at_min) * np.exp(self.ego_vehicle.at - self.acc_param.at_min))
        if self.ego_vehicle.jerk >= 0.0:
            reward_comfort_jerk = -2 / (np.exp(self.acc_param.at_max - self.ego_vehicle.jerk) * np.exp(self.acc_param.at_max - self.ego_vehicle.jerk))
        else:
            reward_comfort_jerk = -2 / (np.exp(self.ego_vehicle.jerk - self.acc_param.jerk_min) * np.exp(self.ego_vehicle.jerk - self.acc_param.jerk_min))
        reward_comfort = np.min([reward_comfort_at, reward_comfort_jerk])

        # 加权奖励
        reward = w_follow * reward_follow + w_safety * reward_safety + w_comfort * reward_comfort

        return reward