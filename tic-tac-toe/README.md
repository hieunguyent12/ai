uses Q-learning to train itself

just some notes

markov decision process:

- a framework used to model decision-making situations in which the outcomes might be random, and the actions that lead to these outcomes might be rewarded.
- formally describe the observable environment for reinforcement learning

From the book:

Consider playing Tic-Tac-Toe against an opponent who plays randomly. In particular, assume the opponent chooses with uniform probability any open space, unless there is a forced move (in which case it makes the obvious correct move).

Formulate the problem of learning an optimal Tic-Tac-Toe strategy in this case as a Q-learning task. What are the states, transitions, and rewards in this non-deterministic Markov decision process?

states: number of x and o's on the board. where each x and o are located
for example
[o, _, x]
[_, x, _]
[_, _, o]

transitions:
action of placing x's or o's on the board

rewards: winning = 1, losing = -1, draw or continue = 0

Q-learning task
