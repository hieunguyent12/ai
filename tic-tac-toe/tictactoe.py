class Game:
    def __init__(self):
        # fmt: off
        self.board = [
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ]
        # fmt: on
        self.player = 0
        self.winner = None
        self.tie = False

    def move(self, x, y):
        self.board[x][y] = "X" if self.player == 0 else "O"
        self.checkGameOver()
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

    def display(self):
        for row in self.board:
            for cell in row:
                print(cell, end=" ")
            print()


class AI:
    def __init__(self):
        self.q = dict()


def train(n):
    """
    Train the AI to play `n` games against itself using Q-Learning
    """

    ai = AI()

    for i in range(n):
        game = Game()

        # how do we handle the fact that are alot of states to consider?
        # can we do immediate rewards instead of rewards for end states?

        while True:
            pass

    return ai


def main():
    test = dict()
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    test[tuple(board)] = 1

    print(test[tuple(board)])
    # game = Game()

    # while True:
    #     print()

    #     if game.winner is not None:
    #         print(f"game over! player {game.winner} won!")
    #         return
    #     elif game.tie:
    #         print("tie game!")
    #         return

    #     # space separated
    #     move = input("Choose a move: ")
    #     x, y = move.split(" ")
    #     game.move(int(x), int(y))

    #     game.display()


if __name__ == "__main__":
    main()
