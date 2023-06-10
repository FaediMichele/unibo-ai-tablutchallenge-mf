from games.tablut.game import State, Action
from collections import defaultdict

class TransferableMemory:
    def __init__(self) -> None:
        self.cache = []
        self.N: dict[tuple[State, Action], int] = defaultdict(lambda: 0)
        self.W: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
        self.Q: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
        self.P: dict[tuple[State, Action], float] = dict()
        self.parent_child: dict[tuple[State, Action], State] = dict()
        self.state_actions: dict[State, list[Action]] = dict()
        self.explored_branch: dict[State, bool] = defaultdict(lambda: False)


    def reset(self):
        self.N = defaultdict(lambda: 0)
        self.W = defaultdict(lambda: 0.0)
        self.Q = defaultdict(lambda: 0.0)
        self.P = dict()
        self.parent_child = dict()
        self.state_actions = dict()
        self.explored_branch = defaultdict(lambda: False)


    def remove_old_branch(self, current_state: State):
        visited_states_for_branch = self._descendant(current_state)

        tmp = defaultdict(lambda: 0)
        for (s,a), v in self.N.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.N = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in self.W.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.W = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in self.Q.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.Q = tmp

        tmp = defaultdict(lambda: 0.0)
        for s, v in self.explored_branch.items():
            if s in visited_states_for_branch:
                tmp[s] = v
        self.explored_branch = tmp

        self.P = dict(((s, a), v) for (s,a), v in self.P.items()
                 if s in visited_states_for_branch)
        
        self.parent_child = dict(
            ((s, a), v) for (s, a), v in self.parent_child.items()
            if s in visited_states_for_branch)
        self.state_actions = dict((s,v) for s,v in
                                         self.state_actions.items()
                                         if s in visited_states_for_branch)


    def _descendant(self, state) -> list[State]:
        if state in self.state_actions.keys():
            return list(self._descendant(child)
                        for a in self.state_actions[state]
                        for child in self.parent_child[(state, a)]
                        ) + [state]
        else:
            return [state]