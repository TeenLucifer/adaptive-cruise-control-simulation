import matplotlib.axes
import numpy as np
import matplotlib

class VehicleParam():
    def __init__(self, length, width, height, wheelbase, front_wheel_to_bumper, rear_wheel_to_bumper):
        self.length    = length    # 车长
        self.width     = width     # 车宽
        self.height    = height    # 车高
        self.wheelbase = wheelbase # 轴距
        self.front_wheel_to_bumper = front_wheel_to_bumper # 前悬长度
        self.rear_wheel_to_bumper  = rear_wheel_to_bumper  # 后悬长度

        self.L1 = self.rear_wheel_to_bumper # 后轴中心到车位距离
        self.L2 = self.length - self.L1     # 后轴中心到车头距离

'''
以后轴中心为原点的车辆模型
'''
class BaseVehicle:
    def __init__(self, vehicle_param: VehicleParam):
        # 车辆参数信息
        self.vehicle_param = vehicle_param
        self.jerk = 0.0
        self.at = 0.0
        self.v = 0.0
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.dt = 0.01

        self.veh_outline_line = None
        self.veh_innerline_line = None
        self.veh_center_line = None

    def step(self, delta, acc):
        self.jerk = (acc - self.at) / self.dt
        self.at = acc
        delta_v = acc * self.dt
        x_dot   = (self.v + 0.5 * delta_v) * np.cos(self.psi)
        y_dot   = (self.v + 0.5 * delta_v) * np.sin(self.psi)
        psi_dot = (self.v + 0.5 * delta_v) * np.tan(delta) / self.vehicle_param.wheelbase
        self.v  = self.v + delta_v

        self.x = self.x + x_dot * self.dt
        self.y = self.y + y_dot * self.dt
        self.psi = self.psi + psi_dot * self.dt
        self.psi = self.normalize_angle(self.psi)

    def draw(self, axes: matplotlib.axes.Axes, color: str):
        # 车辆轮廓
        vehicle_outline = np.array([
            [-self.vehicle_param.L1,
             self.vehicle_param.L2,
             self.vehicle_param.L2,
             -self.vehicle_param.L1,
             -self.vehicle_param.L1],
            [self.vehicle_param.width / 2,
             self.vehicle_param.width / 2,
             -self.vehicle_param.width / 2,
             -self.vehicle_param.width / 2,
             self.vehicle_param.width / 2]])

        # 后轴
        vehicle_innerline = np.array([
            [0, 0],
            [self.vehicle_param.width / 2, -self.vehicle_param.width / 2]
        ])

        # 后轴中心
        rear_center = np.array([[0], [0]])

        # 车身可视化坐标变换
        rot2d = np.array([[np.cos(self.psi), -np.sin(self.psi)],
                          [np.sin(self.psi), np.cos(self.psi)]])
        pos = np.array([[self.x], [self.y]])
        vehicle_outline = np.dot(rot2d, vehicle_outline) + pos
        vehicle_innerline = np.dot(rot2d, vehicle_innerline) + pos
        rear_center = rear_center + pos

        if (self.veh_outline_line is None) or (self.veh_innerline_line is None) or (self.veh_center_line is None):
            self.veh_outline_line,   = axes.plot(vehicle_outline[0, :],   vehicle_outline[1, :], color=color, linewidth=2)
            self.veh_innerline_line, = axes.plot(vehicle_innerline[0, :], vehicle_innerline[1, :], color=color, linewidth=2)
            self.veh_center_line,    = axes.plot(rear_center[0], rear_center[1], 'o', color='tab:red', markersize=4)
        else:
            self.veh_outline_line.set_data(vehicle_outline[0, :], vehicle_outline[1, :])
            self.veh_innerline_line.set_data(vehicle_innerline[0, :], vehicle_innerline[1, :])
            self.veh_center_line.set_data(rear_center[0], rear_center[1])

    def set_state(self, x, y, psi, v):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v

    def normalize_angle(self, angle):
        a = np.fmod(angle + np.pi, 2 * np.pi)
        if a < 0.0:
            a += (2.0 * np.pi)
        return a - np.pi
class EgoVehicle(BaseVehicle):
    pass

class ObjVehicle(BaseVehicle):
    pass