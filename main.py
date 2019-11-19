import gym

if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # random action
    env.close()