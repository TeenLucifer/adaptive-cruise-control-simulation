import matplotlib.axes
import numpy as np

class Lane:
    def __init__(self, length, width):
        self.width = width
        self.length = length
        self.marker_step = 50.0

        left_lane_x  = np.linspace(0, self.length, round(self.length / self.marker_step))
        left_lane_y  = np.full_like(left_lane_x, self.width / 2)
        mid_lane_x   = np.linspace(0, self.length, round(self.length / self.marker_step))
        mid_lane_y   = np.full_like(mid_lane_x, 0)
        right_lane_x = np.linspace(0, self.length, round(self.length / self.marker_step))
        right_lane_y = np.full_like(right_lane_x, -self.width / 2)

        self.left_lane  = np.array([left_lane_x, left_lane_y])
        self.mid_lane   = np.array([mid_lane_x, mid_lane_y])
        self.right_lane = np.array([right_lane_x, right_lane_y])

        self.left_line  = None
        self.mid_line   = None
        self.right_line = None

    def draw(self, axes: matplotlib.axes.Axes, lane_range: np.array):
        if (self.left_line is None) or (self.mid_line is None) or (self.right_line is None):
            self.left_line,  = axes.plot(self.left_lane[0], self.left_lane[1], color='k', linestyle='-')
            self.mid_line,   = axes.plot(self.mid_lane[0], self.mid_lane[1], color='k', linestyle='--', marker='>')
            self.right_line, = axes.plot(self.right_lane[0], self.right_lane[1], color='k', linestyle='-')
            axes.set_xlim(lane_range[0], lane_range[1])
        else:
            axes.set_xlim(lane_range[0], lane_range[1])