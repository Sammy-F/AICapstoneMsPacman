# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, time
import numpy as np

from game import Agent

class LearningAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    rlTable = np.zeros((4, 4))
    learningRate = .1
    alpha = .5
    def getAction(self, gameState):
        """
        Look at legal moves and where direction is to get next move from the table
        then update the table using hte reward from that action
        """
        # Collect legal moves and action states
        legalMoves = gameState.getLegalActions()
        actions = ['North', 'East', 'South', 'West']

        # get info about game
        ghostDir = gameState.getGhostDirections()
        possActions = self.legalToIndices(legalMoves)

        # pick action
        action, ghostDirIndex = self.pickMaxAction(ghostDir, possActions)

        if not action:
            dir = random.choice(legalMoves)
            action = actions.index(dir)

        # q-learning
        reward = self.calcReward(gameState, actions[action])
        self.updateTable(ghostDirIndex, action, reward)

        print(self.rlTable)

        return actions[action]

    def calcReward(self, gameState, action):
        """ Looking at the score difference before / after action is taken, adjust
        the reward accordingly """
        scoreDiff = gameState.generatePacmanSuccessor(action).getScore() - gameState.getScore()
        # if you run into ghost, reward is -50
        if scoreDiff == -501.0:
            return -50.0

        # if you stay alive, reward is 5.0
        elif scoreDiff == -1.0:
            return 5.0

        # if you get pellet, reward is 5.0
        elif scoreDiff == 9.0:
            return 5.0

        return 5.0

    def updateTable(self, ghostDir, action, reward):
        """ Update the reinforcement learning table according to where on the table
        action was from and what end reward was """
        if not ghostDir:
            return
        oldVal = self.rlTable[ghostDir][action]
        self.rlTable[ghostDir][action] = (1-self.alpha)*oldVal + self.learningRate * (reward)

    def pickMaxAction(self, ghostDir, legalIndices):
        """ Look at all directions ghosts are in and in table, find action w/ highest
        reward value for that ghost direction """
        action = None
        ghostDirI = None
        actionVal = 0
        for i in range(len(ghostDir)):
            if ghostDir[i]:
                for j in range(len(legalIndices)):

                    if legalIndices[j]:
                        tempActionVal = self.rlTable[i][j]
                        if tempActionVal >= actionVal:
                            actionVal = tempActionVal
                            action = j
                            ghostDirI = i
                            print(i, j)

                # tempAction = np.argmax(self.rlTable[i])
                # if legalIndices[tempAction]:
        return action, ghostDirI

    def legalToIndices(self, legalMoves):
        """ make list where True corresponds to direction it is legal for pacman
        to move in """
        legal = [False, False, False, False]
        directions = ['North', 'East', 'South', 'West']
        for i in range(len(directions)):
            if directions[i] in legalMoves:
                legal[i] = True
        return legal

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
