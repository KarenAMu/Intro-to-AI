# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from searchAgents import PositionSearchProblem

    #dictionary to keep track of visited states - key: state, value: boolean to indicate visited or not
    visited = {}
    fringe = util.Stack()

    #initialize stack with start node and empty path
    #stack's internal data structure is a list of size 2
    #list[0] is current state, list[1] is path from start to that state
    initialize = []
    nullPath = []
    initialize.append(problem.getStartState())
    initialize.append(nullPath)

    #set start state as not visited
    visited[problem.getStartState()] = False

    fringe.push(initialize)

    while True:
        if fringe.isEmpty():
            return -1

        curr = fringe.pop()

        #get current state that we will try to explore
        prev = curr[0]

        #get path from start state to current state
        path = curr[1]
        isVisited = False

        #return path if state is Goal state
        if problem.isGoalState(prev):
            return path

        #check if we should explore state
        if prev in visited:
            isVisited = visited[prev]
        else:
            visited[prev] = False

        #if state is not visited, mark as visited and add all successors to fringe
        if isVisited == False:
            visited[prev] = True
            allStates = problem.getSuccessors(prev)
            for state in allStates:
                temp = []
                temp.append(state[0])

                #update path with "child" path and current direction of state
                newPath = path + [state[1]]
                temp.append(newPath)
                fringe.push(temp)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    from searchAgents import PositionSearchProblem

    #dictionary to keep track of visited states - key: state, value: boolean to indicate visited or not
    visited = {}
    fringe = util.Queue()

    #initialize stack with start node and empty path
    #queue's internal data structure is a list of size 2
    #list[0] is current state, list[1] is path from start to that state
    initialize = []
    nullPath = []
    initialize.append(problem.getStartState())
    initialize.append(nullPath)

    #set start state as not visited
    visited[problem.getStartState()] = False

    fringe.push(initialize)

    while True:
        if fringe.isEmpty():
            return -1

        curr = fringe.pop()

        #get current state that we will try to explore
        prev = curr[0]

        #get path from start state to current state
        path = curr[1]
        isVisited = False

        #return path if state is Goal state
        if problem.isGoalState(prev):
            return path

        #check if we should explore state
        if prev in visited:
            isVisited = visited[prev]
        else:
            visited[prev] = False

        #if state is not visited, mark as visited and add all successors to fringe
        if isVisited == False:
            visited[prev] = True
            allStates = problem.getSuccessors(prev)
            for state in allStates:
                temp = []
                temp.append(state[0])

                #update path with "child" path and current direction of state
                newPath = path + [state[1]]
                temp.append(newPath)
                fringe.push(temp)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    from searchAgents import PositionSearchProblem

    #dictionary to keep track of visited states - key: state, value: boolean to indicate visited or not
    visited = {}
    fringe = util.PriorityQueue()

    #initialize stack with start node, empty path, and 0 cost
    #priority queue's internal data structure is a list of size 3
    #list[0] is current state, list[1] is path from start to that state, list[2] is cost
    initialize = []
    nullPath = []
    initialize.append(problem.getStartState())
    initialize.append(nullPath)
    initialize.append(0)

    #set start state as not visited
    visited[problem.getStartState()] = False

    #push list to fringe and give priority for lesser costs
    fringe.push(initialize,initialize[2])

    while True:
        if fringe.isEmpty():
            return -1

        curr = fringe.pop()

        #get current state that we will try to explore
        prev = curr[0]

        #get path from start state to current state
        path = curr[1]

        #get cost from root to that state
        cost = curr[2]
        isVisited = False

        #return path if state is Goal state
        if problem.isGoalState(prev):
            return path

        #check if we should explore state
        if prev in visited:
            isVisited = visited[prev]
        else:
            visited[prev] = False

        #if state is not visited, mark as visited and add all successors to fringe
        if isVisited == False:
            visited[prev] = True
            allStates = problem.getSuccessors(prev)
            for state in allStates:
                temp = []
                temp.append(state[0])

                #update path with "child" path and current direction of state
                newPath = path + [state[1]]
                temp.append(newPath)

                #find cost from root to new state that can be explored
                newCost = problem.getCostOfActions(newPath)
                temp.append(newCost)
                fringe.push(temp,newCost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    from searchAgents import PositionSearchProblem

    #dictionary to keep track of visited states - key: state, value: boolean to indicate visited or not
    visited = {}
    fringe = util.PriorityQueue()

    #initialize stack with start node, empty path, and cost of 0
    #priority queue's internal data structure is a list of size 3
    #list[0] is current state, list[1] is path from start to that state
    #list[2] is cost + cost computed from given heuristic
    initialize = []
    nullPath = []
    initialize.append(problem.getStartState())
    initialize.append(nullPath)
    initialize.append(0)

    #set start state as not visited
    visited[problem.getStartState()] = False

    #push list to fringe and give priority for lesser costs
    fringe.push(initialize,heuristic(problem.getStartState(), problem))

    while True:
        if fringe.isEmpty():
            return -1

        curr = fringe.pop()

        #get current state that we will try to explore
        prev = curr[0]

        #get path from start state to current state
        path = curr[1]

        #get cost from root to that state
        cost = curr[2]
        isVisited = False

        #return path if state is Goal state
        if problem.isGoalState(prev):
            return path

        #check if we should explore state
        if prev in visited:
            isVisited = visited[prev]
        else:
            visited[prev] = False

        #if state is not visited, mark as visited and add all successors to fringe
        if isVisited == False:
            visited[prev] = True
            allStates = problem.getSuccessors(prev)
            for state in allStates:
                temp = []
                temp.append(state[0])

                #update path with "child" path and current direction of state
                newPath = path + [state[1]]
                temp.append(newPath)

                #find cost from root to new state that can be explored
                #cost is computed as: cost from root to new state + same path computed by heuristic
                newCost = problem.getCostOfActions(newPath) + heuristic(state[0], problem)
                temp.append(newCost)
                fringe.push(temp,newCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
