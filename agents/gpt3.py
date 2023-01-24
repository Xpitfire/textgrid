from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import openai
from .base import Agent
import re
import sys


class GPT3Agent(Agent):
    def __init__(self, env, model_id):
        super().__init__(model_id, env)
        openai.api_key = "<OPENAI_API_KEY>"
        self.model = model_id

    def action(self, state) -> str:
        did_replace = False
        # check if list of states or single state
        if not isinstance(state, list):
            prompt = state
        else:
            # check if Nones in list
            if None not in state:
                prompt = state
            else:
                # if None in list, replace and keep track via did_replace
                did_replace = True
                idx_map = {}
                new_state = []
                for i, s in enumerate(state):
                    if s is not None:
                        new_state.append(s)
                        idx_map[i] = len(new_state) - 1
                    else:
                        idx_map[i] = None
                prompt = new_state
        
        # send prompt to GPT-3
        res = openai.Completion.create(model=self.model,
                                       prompt=prompt,
                                       max_tokens=1,
                                       temperature=0.8,
                                       frequency_penalty=0,
                                       presence_penalty=0,
                                       logprobs=4,
                                       stop=[',  ;;'],
                                       top_p=1)
        
        # get action from GPT-3
        actions = []
        for choice in res['choices']:
            output = choice['text'].strip()
            if len(output) > 0:
                action = output[0]
            else:
                self.blank_actions += 1
                action = self.env.sample()
            actions.append(action)
            
        # verify actions if valid
        actions, valid = self.env.validate_action(actions)
        self.invalid_action_cnt += len(actions) - sum([int(v) for v in valid])
        
        # fix back None replacement to match previous indices
        if did_replace:
            new_actions = []
            for i, j in idx_map.items():
                if j is not None:
                    new_actions.append(actions[j])
                else:
                    new_actions.append(None)
            actions = new_actions
        # return single or list of actions
        return actions if isinstance(state, list) else actions[0]

    def list_models() -> List:
        return openai.Model.list()
