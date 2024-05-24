import math
import random
import matplotlib.pyplot as plt

GRAPH_SMOOTHING = 5


class Game:
    def __init__(self):
        # fmt: off
        self.board = [
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ]
        # fmt: on
        self.player = 0  # 0 = X, 1 = O
        self.winner = None
        self.tie = False
        self.isOver = False

    # location is a tuple (x, y)
    def move(self, location):
        x, y = location
        self.board[x][y] = "X" if self.player == 0 else "O"
        if self.checkGameOver():
            self.isOver = True

        self.switch_player()

    def checkGameOver(self):
        # diagonals
        if (
            self.board[0][0] == self.board[1][1] == self.board[2][2] != 0
            or self.board[0][2] == self.board[1][1] == self.board[2][0] != 0
        ):
            self.winner = self.player
            return True

        # vertical and horizontal
        for x in range(0, 3):
            if (
                self.board[x][0] == self.board[x][1] == self.board[x][2] != 0
                or self.board[0][x] == self.board[1][x] == self.board[2][x] != 0
            ):
                self.winner = self.player
                return True

        # if every cell is occupied, its a tie
        for row in self.board:
            for item in row:
                if item == 0:
                    return False

        self.tie = True
        return True

    def switch_player(self):
        self.player = 0 if self.player == 1 else 1

    def get_state(self):
        """
        returns a string representation of the board
        """
        state = ""
        for row in self.board:
            for item in row:
                state += str(item)
        return state

    def display(self):
        for row in self.board:
            for cell in row:
                print(cell, end=" ")
            print()

    @classmethod
    def other_player(cls, player):
        return 1 if player == 0 else 0

    @classmethod
    def parse_state(cls, state):
        """
        return the actual 2D array representation of the board from a string state
        """
        board = []
        row = []
        count = 1
        for cell in state:
            row.append(int(cell) if cell == "0" else cell)

            if count % 3 == 0:
                board.append(row)
                row = []
                count = 1
            else:
                count += 1

        return board

    @classmethod
    def get_actions(cls, state):
        """
        returns a list of unoccupied cells
        """

        locations = []
        board = Game.parse_state(state)
        for x in range(0, 3):
            for y in range(0, 3):
                if board[x][y] == 0:
                    locations.append((x, y))

        return locations


class AI:
    def __init__(self, alpha=0.5, epsilon=0.2):
        self.q = dict()
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # probability of exploration

        # initialize empty q for each player?
        self.q[0] = dict()
        self.q[1] = dict()

    def update(self, old_state, action, player, new_state, reward):
        current = self.get_q_value(old_state, action, player)
        future = self.best_future_reward(new_state, player)
        self.update_q_value(old_state, action, current, future, reward, player)

    def get_q_value(self, state, action, player):
        return (
            self.q[player][(state, action)] if (state, action) in self.q[player] else 0
        )

    def update_q_value(self, state, action, current, future, reward, player):
        self.q[player][(state, action)] = current + self.alpha * (
            (future + reward) - current
        )

    def best_future_reward(self, new_state, player):
        actions = Game.get_actions(new_state)
        highest_q_value = 0
        for action in actions:
            highest_q_value = max(
                highest_q_value, self.get_q_value(new_state, action, player)
            )

        return highest_q_value

    def pick_action(self, state, player, training=False):
        """
        given a list of actions, pick a random action with probabilty of epsilon, otherwise
        pick the action that yields the best outcome/reward
        """
        actions = Game.get_actions(state)
        if (not self.q) or (training and random.random() <= self.epsilon):
            return random.sample(actions, 1)[0]

        # otherwise, pick the action with the highest q value
        highest_q_value = -math.inf
        best_action = random.sample(actions, 1)[0]

        for action in actions:
            cur = self.get_q_value(state, action, player)
            if cur > highest_q_value:
                highest_q_value = cur
                best_action = action

        return best_action


class RandomOpponent:
    def __init__(self):
        return

    def get_move(self, state):
        return random.sample(Game.get_actions(state), 1)[0]


def train(n):
    """
    Train the AI to play `n` games against itself using Q-Learning
    """

    ai = AI()

    for _ in range(n):
        game = Game()

        # how do we handle the fact that are alot of states to consider?
        # can we do immediate rewards instead of rewards for end states?

        # last = {0: {state: None, action: None}, 1: {state: None, action: None}}
        prevPlayer = {"state": None, "action": None}

        while not game.isOver:
            state = game.get_state()
            action = ai.pick_action(state, game.player, training=True)

            # last[game.player]["state"] = state
            # last[game.player]["action"] = action

            game.move(action)
            new_state = game.get_state()

            if game.isOver:
                if game.winner is not None:
                    # reward the winning move
                    ai.update(state, action, game.winner, new_state, 1)
                    # punish the bad move that led to the opponent winning
                    ai.update(
                        prevPlayer["state"],
                        prevPlayer["action"],
                        Game.other_player(game.winner),
                        new_state,
                        -1,
                    )
                elif game.tie:
                    ai.update(state, action, game.player, new_state, 0.5)
            else:
                prevPlayer["state"] = state
                prevPlayer["action"] = action
                ai.update(state, action, game.player, new_state, 0)

    return ai


def trial():
    opponent = RandomOpponent()
    ai = train(50000)
    numTrials = 100
    q_wins = []

    for _ in range(numTrials):
        game = Game()

        # ai_player = random.randint(0, 1)
        ai_player = 0

        while not game.isOver:
            state = game.get_state()
            action = ai.pick_action(state, ai_player)
            game.move(action)

            if game.isOver:
                if game.winner == ai_player:
                    q_wins.append(1)
                elif game.tie:
                    q_wins.append(0)
            else:
                game.move(opponent.get_move(state))

    # why is the number so low?
    print(f"Length: {len(q_wins)}-{q_wins}")


def main():
    ai = train(10000)
    game = Game()

    print(f"LENGTH: {len(ai.q[1].values())}")
    # ai = AI()

    human = random.randint(0, 1)

    while True:
        print()

        available_actions = Game.get_actions(game.get_state())

        if game.player == human:
            print("Your Turn")
            while True:
                # space separated
                move = input("Choose a move: ")
                x, y = move.split(" ")
                if (int(x), int(y)) in available_actions:
                    break
                print("invalid move, try again")
        else:
            print("AI's Turn")
            x, y = ai.pick_action(game.get_state(), game.player)

        # python doesn't have scoping rules like c++ or js??
        game.move((int(x), int(y)))
        game.display()
        state = game.get_state()
        print(Game.get_actions(state))

        if game.winner is not None:
            print(f"game over! player {game.winner} won!")
            return
        elif game.tie:
            print("tie game!")
            return


if __name__ == "__main__":
    trial()
