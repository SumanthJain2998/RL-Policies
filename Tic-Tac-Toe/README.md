# Tic-Tac-Toe

In this Project an RL agent was trained to play Tic-Tac-Toe with ___Temporal Difference Learning___  
<p align="center">
 <img width="546" alt="Screenshot 2025-01-23 at 11 56 26 PM" src="https://github.com/user-attachments/assets/ef55d98c-e97a-48c7-bf97-352a45278eb4" />
</p> 

<p align="center">
<em>
  A sequence of tic-tac-toe moves. The solid black lines represent the moves taken during a game; the dashed lines represent moves that we (our reinforcement learning player) considered but did not make. The * indicates the move currently estimated to be the best. Our second move was an exploratory move, meaning that it was taken even though another sibling move, the one leading to e⇤, was ranked higher. Exploratory moves do not result in any learning, but each of our other moves does, causing updates as suggested by the red arrows in which estimated values are moved up the tree from later nodes to earlier nodes as detailed in the text.
</em>  
</p>
&nbsp;  

If we let \$S_t\$ denote the state before greedy moove, and \$S_{t+1}\$ the state after that move and then update the estimated value of \$S_t\$, denoted \$V(S_t)\$  
```math
V(S_t) \leftarrow V(S_t)+\alpha\left[V(S_{t+1})-V(S_t)\right]
```
where \$\alpha\$ is a small _step-size_ parameter, which influences the rate of learning. This update rule is an example of a _temporal-difference_ learning method, so called because its changes are based on a difference, \$V(S_{t+1})-V(S_t)\$, between estimates at two successive times

## References
Sutton, Richard S., Barto, Andrew G. "Reinforcement Learning: An Introduction." United Kingdom: MIT Press, 2018.
