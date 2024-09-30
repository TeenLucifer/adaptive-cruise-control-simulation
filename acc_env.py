import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from vehicle_model import VehicleParam, EgoVehicle, ObjVehicle
from lane_model import Lane

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

        # 环境模型
        self.vehicle_param = VehicleParam(4.935, 1.915, 1.495, 2.915, 1.042, 1.191) # 车辆参数
        self.ego_vehicle = EgoVehicle(self.vehicle_param) # 主车
        self.obj_vehicle = ObjVehicle(self.vehicle_param) # 目标
        self.lane = Lane(width=10.0, length=10000.0)      # 车道

        # 可视化
        self.fig = plt.figure('ACC Simulaion')                    # 画布
        self.vis_ax = self.fig.add_axes([0.03, 0.03, 0.93, 0.93]) # 绘图坐标轴

    def reset(self):
        self.state = np.zeros(self.state.shape[0], dtype=np.float32)
        self.done  = False

    def step(self, action):
        # TODO(wangjintao): 更新小车状态
        ego_delta = 0.0
        obj_delta = 0.0
        ego_acc = action
        obj_acc = 0.0
        self.ego_vehicle.step(ego_delta, ego_acc)
        self.obj_vehicle.step(obj_delta, obj_acc)

        # TODO(wangjintao): 判断是否结束(失败或成功都结束)
        # TODO(wangjintao): 获取reward
        reward = 0.0
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        plt.axis('equal')
        self.lane.draw(self.vis_ax, [self.ego_vehicle.x - 20, self.ego_vehicle.x + 100])
        self.ego_vehicle.draw(self.vis_ax, 'tab:blue')
        self.obj_vehicle.draw(self.vis_ax, 'tab:green')
        self.fig.canvas.draw_idle()    # 重绘图形
        self.fig.canvas.flush_events() # 刷新

    def reward_funcion(self):
        pass