import argparse
import enum
import os
from typing import Dict, Tuple, List
from abc import ABC
from utils import Entry, MultiEpisode, Stats


class RunStatus(enum.Enum):
    """Status of the run."""
    INIT = 0
    RUNNING = 1
    FINISHED = 3


class MultiEnv(ABC):
    def __init__(self, envs: List, max_runs: int, video: bool = False, video_descr: str = ''):
        self.envs = envs
        self.run_status = {i: RunStatus.INIT for i in range(len(envs))}
        self.max_runs = max_runs
        self.run_cnt = 0
        self.video = video
        self.video_descr = video_descr
        self.stats = Stats()
        self.stats.n_runs = max_runs
        self.n_runs = max_runs
        
    @property
    def action_space(self):
        return self.envs[0].action_space
        
    @property
    def all_fails(self):
        return self.died + self.failed
    
    @property
    def done(self) -> bool:
        return not any([self.run_status[i] == RunStatus.RUNNING for i in range(len(self.envs))])
    
    def reset(self, *args, **kwargs) -> object:
        self.run_cnt = len(self.envs)
        for i, env in enumerate(self.envs):
            self.run_status[i] = RunStatus.RUNNING
        return [env.reset(*args, **kwargs) for env in self.envs]
    
    def step(self, actions: List) -> List:
        steps = []
        for i, env in enumerate(self.envs):
            if self.run_status[i] == RunStatus.RUNNING:
                state, reward, done, info = env.step(actions[i])
                if done:
                    if 'Congrats' in state:
                        self.stats.n_success += 1
                        self.stats.episode_lengths.append(self.envs[i].timestep)
                        self.stats.optimal_episode_lengths.append(len(self.envs[i].path)-1)
                        if self.video:
                            env.write_video(f'{self.video_descr}_success')
                            env.save_action_view(f'{self.video_descr}_success')
                    elif 'died' in state:
                        self.stats.n_died += 1
                        self.stats.n_non_success += 1
                        if self.video: 
                            env.write_video(f'{self.video_descr}_died')
                            env.save_action_view(f'{self.video_descr}_died')
                    else:
                        self.stats.n_failed += 1
                        self.stats.n_non_success += 1
                        if self.video: 
                            env.write_video(f'{self.video_descr}_failed')
                            env.save_action_view(f'{self.video_descr}_failed')

                    if self.run_cnt >= self.max_runs:
                        self.run_status[i] = RunStatus.FINISHED
                    elif self.run_status[i] == RunStatus.RUNNING:
                        state = env.reset()
                        self.run_cnt += 1
                steps.append((state, reward, done, info))
            else:
                steps.append((None, None, None, None))
        return [s[0] for s in steps], [s[1] for s in steps], [s[2] for s in steps], [s[3] for s in steps]
    
    def prepare_state(self, states: str, *args, **kwargs) -> List:
        return [env.prepare_state(state, *args, **kwargs) if state is not None else None for env, state in zip(self.envs, states)]
    
    def write_video(self, path: str):
        return [env.write_video(path) for env in self.envs]
    
    def validate_action(self, actions: List) -> Tuple[List, List[bool]]:
        actions = [env.validate_action(action) for env, action in zip(self.envs, actions)]
        return [a[0] for a in actions], [a[1] for a in actions]
    
    def sample(self) -> str:
        actions = []
        for i, env in enumerate(self.envs):
            if self.run_status[i] == RunStatus.RUNNING:
                actions.append(env.sample())
            else:
                actions.append(None)
        return actions
    
    def optimal(self) -> str:
        actions = []
        for i, env in enumerate(self.envs):
            if self.run_status[i] == RunStatus.RUNNING:
                actions.append(env.optimal())
            else:
                actions.append(None)
        return actions

    def entries(self, states: List, actions: List, next_states: List, rewards: List, dones: List) -> List[Entry]:
        return [Entry(state, action, next_state, reward, done) for state, action, next_state, reward, done in zip(states, actions, next_states, rewards, dones)]
