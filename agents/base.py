from abc import ABC, abstractmethod
from typing import List


class Agent(ABC):
    def __init__(self, id, env):
        self.id = id
        self.env = env
        self.invalid_action_cnt = 0
        self.blank_actions = 0
    
    @abstractmethod
    def action(self, state: str) -> str:
        pass
    
    def actions(self, states: List[str]) -> List[str]:
        return [self.action(state) for state in states]
    
    
class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__('Random', env)
    
    def action(self, state: str) -> str:
        return self.env.sample()
    
    
class OptimalAgent(Agent):
    def __init__(self, env):
        super().__init__('Optimal', env)
    
    def action(self, state: str) -> str:
        return self.env.optimal()
