from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import random
from .base import Agent
import re
import sys
import symai as ai
from multiprocessing import cpu_count, Pool


class SymAIAgent(Agent):
    def __init__(self, env):
        super().__init__('symai', env)
        self.invalid_action_cnt = 0

    class ValueIteration(ai.Expression):
        @ai.few_shot(
            prompt="""Evaluate the actions surrounding the P character. The pattern is as follows:
Surrounding next to P means in 4 directions (up, down, left, right) from the state position.

State Example 1: 
(10, 10) Grid:\nState 1,\n##########\n#**.....*#\n#.G.....X#\n#........#\n#........#\n#........#\n#.....*P.#\n#..X.....#\n#*.*.....#\n##########,\nGoal Distance 9,\nReward 1, ->
1. Extract next to P: \n#........#\n#.....*P.#\n#..X.....#\n
2. Extract possible move states:
- Up to P: .
- Down to P: .
- Left to P: *
- Right to P: .

State Example 2: 
(10, 10) Grid:\nState 1,\n##########\n#**.....*#\n#.G.....X#\n#........#\n#........#\n#........#\n#.....*P.#\n#..X.....#\n#*.*.....#\n##########,\nGoal Distance 9,\nReward 1, ->
1. Extract next to P: \n#........#\n#.....*P.#\n#..X.....#\n
2. Extract possible move states:
- Up to P: .
- Down to P: .
- Left to P: *
- Right to P: .


""")
        def forward(self, state):
            pass
        
    def _parse_action(self, text):
        sym = ai.Symbol(text)
        res = sym.extract('Action: * Value: *')
        print('extract action-value:', res)
        if len(res.strip()) == 0:
            print('empty string found', sym)
        ori = res.rank(measure="Value", order='desc')
        print('rank', ori)
        res = ori.extract('get first Action: * Value: *')
        print('extract first action-value', res)
        val = res.extract('get value number')
        print('extract value', val)
        res = ori.extract(f'list of letters where Action: * Value: {res}')
        print('extract letters', res)
        res = [a.strip() for a in res.split('|')]
        if len(res) == 0:
            i = random.randint(0, len(self.env.action_space)-1)
            action = self.env.action_space[i]
            self.invalid_action_cnt += 1
        else:
            action = random.choice(res)
            if action not in self.env.action_space:
                i = random.randint(0, len(self.env.action_space)-1)
                action = self.env.action_space[i]
                self.invalid_action_cnt += 1
        return action
                
    def action(self, states) -> str:
        pool = Pool(cpu_count() - 1)
        actions = pool.map(self._parse_action, states)
        return actions
