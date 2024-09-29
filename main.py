import matplotlib.pyplot as plt

from vehicle_model import VehicleParam, BaseVehicle

def main():
    fig = plt.figure(1)
    plot_axes = fig.add_axes([0.05, 0.05, 0.95, 0.95])
    vehicle_param = VehicleParam(4.935, 1.915, 1.495, 2.915, 1.042, 1.191)
    vehicle = BaseVehicle(vehicle_param)
    vehicle.draw(plot_axes)
    plt.show()

if __name__ == '__main__':
    main()