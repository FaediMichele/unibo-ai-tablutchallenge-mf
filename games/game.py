from typing import Any
Action = Any
State = Any
Board = Any

class Game:
    ''' Abstract class that represent the game rule.
    Does not contain the state but the method that allow to create the decision tree starting from a state
    '''

    def create_root(self, player: str) -> Board:
        ''' Return the initial state of the game '''
        pass

    def actions(self, state: State) -> list[Action]:
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        pass

    def result(self, state: State, action: Action) -> State:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        pass

    def is_terminal(self, state: State) -> bool:
        ''' Return true if a state determine a win or loose

        Keyword arguments:
        state -- the state that is wanted to check'''
        pass

    def utility(self, state: State, player: str) -> float:
        ''' Return the value of this final state to player

        Keyword arguments:
        state -- the state of the game. The current player is ignored
        player -- the player that want to know the value of a final state
        '''
        pass

    def h(self, state: State, player: str) -> float:
        ''' Euristic function'''
        pass

    def get_player_pieces_values(self, player: str) -> list[int]:
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        pass
