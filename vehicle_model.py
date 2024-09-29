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
        self.x = 0.0
        self.y = 0.0
        self.psi = np.pi / 2
        self.delta = 0.0
        self.dt = 0.01

    def draw(self, axes: matplotlib.axes.Axes):
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

        axes.plot(vehicle_outline[0, :], vehicle_outline[1, :], color="tab:blue", linewidth=2)
        axes.plot(vehicle_innerline[0, :], vehicle_innerline[1, :], color="tab:blue", linewidth=2)
        axes.plot(rear_center[0], rear_center[1], 'o', color='tab:red')

class EgoVehicle(BaseVehicle):
    pass

class ObjVehicle(BaseVehicle):
    pass