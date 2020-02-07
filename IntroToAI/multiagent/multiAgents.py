# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

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
    import math
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    currFood = currentGameState.getFood()
    newFoodList = newFood.asList()

    foodDist = 0
    ghostDist = 999999

    if currFood.count() == len(newFood.asList()):
      foodDist = 999999
      for food in newFoodList:
        MH = util.manhattanDistance(food, newPos)
        foodDist = min(MH,foodDist)

    for ghosts in newGhostStates:
      MH = util.manhattanDistance(ghosts.getPosition(), newPos)
      ghostDist = min(MH,ghostDist)
      foodDist += math.sqrt(9 ** (9 - (2 * ghostDist)))

    "*** YOUR CODE HERE ***"
    #eturn successorGameState.getScore()
    return -1 * foodDist

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    #1 make minVal and MaxVal only compute the +inf or the -inf and the < or > part, put remainder in mainDriver
    #3 cleanup variables in MinVal and MaxVal so we don't have to index [1]
    def mainDriver(gameState, agent, depth):

      #once we run out of agents in the layer, we progress onto the next and reset Pacman
      if agent > gameState.getNumAgents() or agent == gameState.getNumAgents():
        agent = 0 #reset to Pacman
        depth += 1

      #terminator
      if (depth == self.depth or gameState.isWin() or gameState.isLose()):
        evalFunc = self.evaluationFunction(gameState)
        fakeTuple = (evalFunc, "evalFunc")
        return fakeTuple
      
      #extra cautious steps
      actionsList = gameState.getLegalActions(agent) 
      
      vibeCheck(gameState, agent, actionsList)

      return minMax(gameState, agent, depth)


    def vibeCheck(gameState, agent, actionsList):  
      if not actionsList:
        evalFunc = self.evaluationFunction(gameState)
        fakeTuple = (evalFunc, "evalFunc")
        return fakeTuple

    #THIS MIGHT BREAK EVERYTHING
    def minMax(gameState, agent, depth):

      actionsList = gameState.getLegalActions(agent)

      vibeCheck(gameState, agent, actionsList)

      if agent == 0:
        flag = "pacman"
      else:
        flag = "ghost"
        
      if flag == "pacman":
        ret = (-999999, "dicks")

        for action in actionsList:
          nextState = gameState.generateSuccessor(agent, action)
          decidingVal = mainDriver(nextState, agent+1, depth)
        
          returnVal = decidingVal[0]
          comparison = ret[0]
        
          if returnVal > comparison:
            tempTuple = (returnVal, action)
            ret = tempTuple   

        return ret

      if flag == "ghost":
        ret = ("dicks", 999999)

        for action in actionsList:
          nextState = gameState.generateSuccessor(agent, action)
          decidingVal = mainDriver(nextState, agent+1, depth)
        
          returnVal = decidingVal[0]
          comparison = ret[0]
        
          if returnVal < comparison:
            tempTuple = (returnVal, action)
            ret = tempTuple       
        return ret
                    
    return mainDriver(gameState, 0, 0)[1]               
  
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

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
