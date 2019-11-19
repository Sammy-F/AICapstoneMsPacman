import gym
import random
import numpy as np
from gym import spaces
from collections import deque

random.seed(0)

"""
The Bellman equation incorporates two ideas that are highly relevant to this project. 
First, taking a particular action from a particular state many times will result in a 
good estimate for the Q-value associated with that state and action. To this end, you 
will increase the number of episodes this bot must play through in order to return a 
stronger Q-value estimate. Second, rewards must propagate through time, so that the 
original action is assigned a non-zero reward. This idea is clearest in games with 
delayed rewards; for example, in Space Invaders, the player is rewarded when the 
alien is blown up and not when the player shoots. However, the player shooting is 
the true impetus for a reward. Likewise, the Q-function must assign (state0, shoot) 
a positive reward.
"""
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNAgent:
    def __init__(self, observationSpace, actionSpace):
        self.explorationRate = EXPLORATION_MAX
        self.actionSpace = actionSpace
        self.memory = deque(maxlen=MEMORY_SIZE)

class SammyBot:

    def __init__(self, spaceSize=100800):
        self.spaceSize = spaceSize

    def discretize(self, space):
        return space.reshape(self.spaceSize)

    def run(self, numEpisodes=500, seed=0, discountFactor=0.8, learningRate=0.9):
        env = gym.make('MsPacman-v0')
        env.seed(seed)

        observationSpace = env.observation_space.shape[0]
        actionSpace = env.action_space.new
        dqnSolver = DQNSolver(observationSpace, actionSpace)

        # Game loop
        while True:
            state = env.reset()
            state = np.reshape(state, [1, observationSpace])

            while True:
                env.render()
                action = dqnSolver.act(state)
                stateNext, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                stateNext = np.reshape(stateNext, [1, observationSpace])
                dqnSolver.remember(state, action, reward, stateNext, terminal)
                dqnSolver.experience_replay()
                state = stateNext
                if terminal:
                    break

        # Game loop
        for episode in range(1, numEpisodes+1):
            state = self.discretize(env.reset())
            episodeReward = 0
            while True:

                # Add noise to encourage some randomness
                # As episode increases, noise decreases quadratically
                noise = np.random.random((1, env.action_space.n)) / (episode**2.)

                action = np.argmax(Q[state, :] + noise)
                state2, reward, done, _ = env.step(action)
                state2 = self.discretize(state2)

                # Compute new target Q-value
                Qtarget = reward + discountFactor * np.max(Q[state2, :])
                # Update Q-table
                Q[state, action] = (1-learningRate)*Q[state,action] + learningRate*Qtarget


                episodeReward += reward
                state = state2
                if done:
                    rewards.append(episodeReward)
                    break

        print('Average reward: %.2f' % (sum(rewards)/len(rewards)))
        env.close() 

if __name__ == "__main__":
    sBot = SammyBot()
    sBot.run()
