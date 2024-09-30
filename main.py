import matplotlib.pyplot as plt

from acc_env import AccEnv

def main():
    plt.ion()
    env = AccEnv()
    state = env.reset()
    done = False
    while not done:
        state, reward, done, _ = env.step(10.0)
        env.render()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()