# -*- coding: utf-8 -*-
'''
#Tic-Tac-Toe

Using Reinforcement algorithm train an agent to play tic-tac-toe.   


*   Firstly we'll train 2 agents to play against each other and save their policy

*   Use the policy to play against human

refernce:https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
'''

#State Setting
'''
We need a "State" class to act as both board and judger. 
It has functions recording board state of both players and update state when either player takes an action. 
Meanwhile, it is able to judge the end of game and give reward to players accordingly.
'''

cd /content/drive/MyDrive/Colab Notebooks/RL

import numpy as np
import pickle

# Define shape of the board

BOARD_ROWS = 3
BOARD_COLUMNS=3

'''
To formulate this reinforcement learning problem, 
the most important thing is to be clear about the 3 major components 
— state, action, and reward. The state of this game is the board state of both the agent and its opponent, 
so we will initialise a 3x3 board with zeros indicating available positions and update positions with 1 if player 1 takes a move and -1 if player 2 takes a move. 
The action is what positions a player can choose based on the current board state. 
Reward is between 0 and 1 and is only given at the end of the game
'''

# define the class state that acts as board and judger

class State:
  def __init__(self,p1,p2):
    self.board = np.zeros((BOARD_ROWS,BOARD_COLUMNS)) # initialize the board with zeros
    self.p1 = p1                                      # player 1(agent1)
    self.p2 = p2                                      # player 2(agent2)
    self.isEnd = False
    self.boardHash = None
    # init p1 plays first
    self.playerSymbol = 1

#get the unique hash of current board
#The getHash function hashes the current board state so that it can be stored in the state-value dictionary.

  def getHash(self):
    self.boardHash = str(self.board.reshape(BOARD_COLUMNS*BOARD_ROWS))
    return self.boardHash

#When a player takes an action, its corresponding symbol will be filled in the board.
#And after the state being updated, the board will also update the current vacant positions on the board
#and feed it back to the next player in turn.

  def availablePositions(self):
    positions = []
    for i in range (BOARD_ROWS):
      for j in range (BOARD_COLUMNS):
        if self.board[i,j] ==0:
          positions.append((i,j))                     # need to be a tuple
    return positions


  def updateState(self,positions):
    self.board[positions] = self.playerSymbol
    #switch to another player
    self.playerSymbol = -1 if self.playerSymbol==1 else 1

#Check winner
#   After each action being taken by the player,
#   we need a function to continuously check if the game has ended and if end,
#   to judge the winner of the game and give reward to both players.
#
#   The winner function checks sum of rows, columns and diagonals,
#   and return 1 if p1 wins, -1 if p2 wins, 0 if draw and None if the game is not yet ended.
#   At the end of game, 1 is rewarded to winner and 0 to loser.
#   One thing to notice is that we consider draw is also a bad end, so we give our agent p1 0.1 reward even the game is tie
#   (one can try out different reward to see how the agents act)

  def winner(self):
    #row
    for i in range(BOARD_ROWS):
      if sum(self.board[i,:])==3:
        self.isEnd = True
        return 1
      if sum(self.board[i,:])==-3:
        self.isEnd = True
        return -1

    #col
    for i in range(BOARD_COLUMNS):
      if sum(self.board[:,i])== 3:
        self.isEnd = True
        return 1
      if sum(self.board[:,i])==-3:
        self.isEnd = True
        return -1

    #diagonal
    diag_sum1 = sum([self.board[i,i] for i in range(BOARD_COLUMNS)])
    diag_sum2 = sum([self.board[i, BOARD_COLUMNS-i-1] for i in range(BOARD_COLUMNS)])
    diag_sum = max(abs(diag_sum1), abs(diag_sum2))
    if diag_sum ==3:
      self.isEnd = True
      if  diag_sum1 == 3 or diag_sum2 ==3:
        return 1
      else:
        return -1

    #tie
    # no available positions

    if len(self.availablePositions())==0:
      self.isEnd = True
      return 0

    #not end
    self.isEnd = False
    return None

# only when the game ends
  def giveReward(self):
    result  = self.winner()

    #backpropagate the reward
    if result ==1:
      self.p1.feedReward(1)
      self.p2.feedReward(0)
    elif result==-1:
      self.p1.feedReward(0)
      self.p2.feedReward(1)
    else:
      self.p1.feedReward(0.1)
      self.p2.feedReward(0.5)

#board reset
  def reset(self):
    self.board = np.zeros((BOARD_ROWS, BOARD_COLUMNS))
    self.boardHash = None
    self.isEnd = True
    self.playerSymbol  =1

  def play(self, rounds=100):
    for i in range(rounds):
      #if i % 100 ==0:
      #  print("Rounds {}".format(i))
      while not self.isEnd:
        #player 1
        positions = self.availablePositions()
        p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
        # take action and update board state
        self.updateState(p1_action)
        board_hash = self.getHash()
        self.p1.addState(board_hash)
        print('agent_1:',self.p1.states_value)
        #check board status if it is end

        win = self.winner()
        if win is not None:
          self.giveReward()
          self.p1.reset()
          self.p2.reset()
          self.reset()
          break


        else:
          #player 2
          postions = self.availablePositions()
          p2_action = self.p2.chooseAction(positions,self.board, self.playerSymbol)
          self.updateState(p2_action)
          board_hash = self.getHash()
          self.p2.addState(board_hash)
          print('agent_2:',self.p2.states_value)


          win = self.winner()
          if win is not None:
            self.giveReward()
            self.p1.reset()
            self.p2.reset()
            self.reset()
            break
    self.p1.savePolicy()
    self.p2.savePolicy()

#play with human
  def play2(self):
    while not self.isEnd:
      #player1
      positions = self.availablePositions()
      p1_action = self.p1.chooseAction(positions,self.board,self.playerSymbol)
      #take action and update the board
      self.updateState(p1_action)
      self.showBoard()
      win = self.winner()
      if win is not None:
        if win==1:
          print(self.p1.name,'wins!')
        else:
          print('tie!')
        self.reset()
        break
      else:
        positions  = self.availablePositions()
        p2_action = self.p2.chooseAction(positions)

        self.updateState(p2_action)
        self.showBoard()
        win = self.winner()
        if win is not None:
          if win==-1:
            print(self.p2.name, 'wins!')
          else:
            print('tie')
          self.reset
          break

  def showBoard(self):
    #p1:x p2:o
    for i in range(0,BOARD_ROWS):
      print('-------------')
      out = '|'
      for j in range(0, BOARD_COLUMNS):
        if self.board[i,j]==1:
          token = 'x'
        if self.board[i,j]==-1:
          token = 'o'
        if self.board[i,j]==0:
          token = ' '
        out += token + ' | '
      print(out)
    print('-------------')

class Player:
  '''In the init function, we keep track of all positions the player's
   been taken during each game in a list self.states and update the corresponding states in self.states_value dict.
   In terms of choosing action, we use ϵ-greedy method to balance between exploration and exploitation.
   Here we set exp_rate=0.3 , which means ϵ=0.3 , so 70% of the time our agent will take greedy action,
   which is choosing action based on current estimation of states-value,
   and 30% of the time our agent will take random action
   '''
  def __init__(self,name,exp_rate=0.1):
    self.name = name
    self.states = []
    self.lr = 0.01
    self.exp_rate = exp_rate
    self.decay_gamma = 0.9
    self.states_value = {}

  def getHash(self, board):
    boardHash = str(board.reshape(BOARD_COLUMNS*BOARD_ROWS))
    return boardHash

  def chooseAction(self, positions, current_board, symbol):
    if np.random.uniform(0,1) <= self.exp_rate:
      idx = np.random.choice(len(positions))
      action = positions[idx]
    else:
      value_max = -999
      for p in positions:
        next_board = current_board.copy()
        next_board[p] = symbol
        next_boardHash = self.getHash(next_board)
        value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
        #print('value', value)
        if value> value_max:
          value_max = value
          action = p

      #print("{} takes action {}".format(self.name, action))
      return action


    #append a hash state
  def addState(self, state):
    self.states.append(state)

    #at the end of the game, back propagate and update states value
    '''the updated value of state t equals the current value of state t
    adding the difference between the value of next state and the value of current state,
    which is multiplied by a learning rate α (Given the reward of intermediate state is 0).
    '''
  def feedReward(self, reward):
    for st in reversed(self.states):
      if self.states_value.get(st) is None:
        self.states_value[st] = 0
      self.states_value[st] +=self.lr*(self.decay_gamma*reward-self.states_value[st])
      reward = self.states_value[st]

    '''the positions of each game is stored in self.states
    and when the agent reach the end of the game,
    the estimates are updated in reversed fashion.
    '''

  def reset(self):
    self.states = []

  def savePolicy(self):
    fw = open('new_policy_'+str(self.name),'wb')
    pickle.dump(self.states_value,fw)
    fw.close()

  def loadPolicy(self,file):
    fr = open(file,'rb')
    self.states_value=pickle.load(fr)
    fr.close()

class HumanPlayer():
  def __init__(self, name):
    self.name = name

  def chooseAction(self, positions):
    while True:
      row = int(input("Input your action row:"))
      col = int(input("Input your action column:"))
      action = (row, col)
      if action in positions:
        return action

  def addState(self, state):
    pass

  def feedReward(self, reward):
    pass

  def reset(self):
    pass

"""###Training"""

if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(10000)

if __name__ == "__main__":
# play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()
