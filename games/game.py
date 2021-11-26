Board = list[list[int]]
State = tuple[str, Board]
Action = tuple[int, int, int, int]


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

    def path_cost(self, c: float, state1: State, action: Action, state2: State) -> float:
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def get_player_pieces_values(self, player: str) -> list[int]:
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        pass
