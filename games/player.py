import numpy as np


class Player:
    ''' Player that take random actions '''

    def __init__(self, make_move, board, game, player):
        """Create a new player tha play randomly

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        """
        self.make_move = make_move
        self.board = board
        self.game = game
        self.player = player
        super(Player, self).__init__()

    def next_action(self, last_action):
        ''' Function called when the opponent take the move and now is the turn of this player 

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        actions = self.game.actions(self.board.state)
        if len(actions) == 0:
            print(f"Draw. The {self.player} can not do any moves")
            self.make_move(None)
        else:
            self.make_move(actions[np.random.randint(0, len(actions))])
