import json
from enum import Enum
import os
from typing import List, Tuple


class PolicyType(Enum):
    Expert: str = 'expert'
    Random: str = 'random'
    Pretrained: str = 'pretrained'
    
    def __eq__(self, other: object) -> bool:
        return self.value == other


class Entry:
    def __init__(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done


class Episode:
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []        
    
    def __add__(self, other: Entry):
        self.states.append(other.state)
        self.actions.append(other.action)
        self.next_states.append(other.state)
        self.rewards.append(other.reward)
        self.dones.append(other.done)
        return self
    
    def __getitem__(self, key: int) -> Entry:
        return Entry(self.states[key], self.actions[key], self.next_states[key], self.rewards[key], self.dones[key])
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __str__(self) -> str:
        return '\n'.join([f'{self.states[i]}' for i in range(len(self))])


class MultiEpisode:
    def __init__(self, n_envs: int):
        self.episodes = [Episode() for _ in range(n_envs)]
    
    def __add__(self, other: List[Entry]):
        for i, episode in enumerate(self.episodes):
            self.episodes[i] = episode + other[i]
        return self
    
    def __getitem__(self, key: int) -> Episode:
        return self.episodes[key]
    
    def __len__(self) -> int:
        return len(self.episodes)


class Stats:
    def __init__(self) -> None:
        self.reset()
        
    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "n_success": self.n_success,
            "n_non_success": self.n_non_success,
            "n_failed": self.n_failed,
            "n_died": self.n_died,
            "episode_lengths": self.episode_lengths,
            "optimal_episode_lengths": self.optimal_episode_lengths,
            "invalid_action_cnt": self.invalid_action_cnt,
            "blank_actions": self.blank_actions
        }
        
    def save(self, file_prefix: str) -> None:
        os.makedirs('results', exist_ok=True)
        i = 0
        file_name = f'results/{file_prefix}_{i}.json'
        while os.path.exists(file_name):
            i += 1
            file_name = f'results/{file_prefix}_{i}.json'
        with open(file_name, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def reset(self, n_runs: int = None) -> None:
        if n_runs is not None:
            self.n_runs = n_runs
        self.n_success = 0
        self.n_non_success = 0
        self.n_failed = 0
        self.n_died = 0
        self.episode_lengths = []
        self.optimal_episode_lengths = []
        self.invalid_action_cnt = 0
        self.blank_actions = 0
