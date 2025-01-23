# Tic-Tac-Toe

In this Project an RL agent was trained to play Tic-Tac-Toe with ___Temporal Difference Learning___  
<p align="center">
  <img width="566" alt="Screenshot 2025-01-23 at 11 08 59 PM" src="https://github.com/user-attachments/assets/7d16e2f3-2306-47a9-a5d8-749fb4ce9a46" />  
</p>  

If we let \$S_t\$ denote the state before greedy moove, and \$S_{t+1}\$ the state after that move and then update the estimated value of \$S_t\$, denoted \$V(S_t)\$  
```math
V(S_t) \leftarrow V(S_t)+\alpha\left[V(S_{t+1})-V(S_t)\right]
```
where \$\alpha\$ is a small _step-size_ parameter, which influences the rate of learning. This update rule is an example of a _temporal-difference_ learning method, so called because its changes are based on a difference, \$V(S_{t+1})-V(S_t)\$, between estimates at two successive times

## References
Sutton, Richard S., Barto, Andrew G. "Reinforcement Learning: An Introduction." United Kingdom: MIT Press, 2018.
