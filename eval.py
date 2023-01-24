import argparse
from time import sleep
from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import json
from pathlib import Path
import os
import ast
from scipy.stats import pearsonr 
from tqdm import tqdm
from agents.base import Agent, OptimalAgent, RandomAgent
from agents.symai import SymAIAgent
from agents.gpt3 import GPT3Agent
from envs.textgrid import GridEnv
from envs.base import MultiEnv
from utils import Entry, MultiEpisode, PolicyType


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--n_runs', type=int, default=50,
                        help='number of runs (default: 50)')
    parser.add_argument('--trap_prob', type=float, default=0.05,
                        help='probability of traps')
    parser.add_argument('--obst_prob', type=float, default=0.2,
                        help='probability of obstacles (rocks or trees)')
    parser.add_argument('--tree_prob', type=float, default=0.00,
                        help='probability of trees')
    parser.add_argument('--policy_type', type=str, default='pretrained',
                        help='define policy to evaluate (default: pretrained)')
    parser.add_argument('--add_description', action=argparse.BooleanOptionalAction,
                        help='add description to states')
    parser.add_argument('--add_value_function', action=argparse.BooleanOptionalAction,
                        help='add value function to description')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,
                        help='add debug flag')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction,
                        help='add video flag')
    parser.add_argument('--domain_shift', action=argparse.BooleanOptionalAction,
                        help='add domain shift to description')
    parser.add_argument('--model_id', type=str, 
                        help='set the name of the model to use')
    parser.add_argument('--grid_sizes', type=lambda a: ast.literal_eval(a), nargs='+',
                        default="[(8, 8), (11, 11), (13, 9), (15, 10), (20, 20)]",
                        help='list of grid sizes to evaluate')
    args = parser.parse_args()
    
    if isinstance(args.grid_sizes[0], list):
        args.grid_sizes = args.grid_sizes[0]
    
    return args


def run(args):
    
    for grid_size in args.grid_sizes:
        
        # limited by GPT-3 API
        n_parallel_jobs = min(20, args.n_runs)

        envs = [GridEnv(grid_size,
                        obst_prob=args.obst_prob,
                        trap_prob=args.trap_prob,
                        tree_prob=args.tree_prob,
                        add_description=args.add_description,
                        add_value_function=args.add_value_function,
                        debug=args.debug,
                        render_image=args.video) for _ in range(n_parallel_jobs)]
        env = MultiEnv(envs, args.n_runs, video=args.video, video_descr=f'{args.policy_type}_descr{args.add_description}_shift{args.domain_shift}')
        
        if args.policy_type == PolicyType.Expert:
            agents: List[Agent] = [OptimalAgent(env)]
        elif args.policy_type == PolicyType.Random:
            agents: List[Agent] = [RandomAgent(env)]
        elif args.policy_type == PolicyType.Pretrained:
            if args.model_id == 'symai':
                agents: List[Agent] = [SymAIAgent(env)]
            else:                
                # if model_id is a file, load it
                if os.path.exists(args.model_id) and Path(args.model_id).is_file():
                    # read json file
                    with open(args.model_id, 'r') as f:
                        model_ids = json.load(f)
                    agents: List[Agent] = [GPT3Agent(env, model_id=data['id']) for data in model_ids['data']]
                # else load model directly
                else:
                    agents: List[Agent] = [GPT3Agent(env, model_id=args.model_id)]
        else:
            raise ValueError('Invalid agent type.')
        
        # run for each agent
        for agent in agents:
            state: str = env.reset()
            episodes: MultiEpisode = MultiEpisode(n_parallel_jobs)
            while not env.done:
                prep_state = env.prepare_state(state, grid_size, args.add_description, args.domain_shift)
                action = agent.action(prep_state)
                next_state, reward, done, _ = env.step(action)
                episodes += env.entries(state, action, next_state, reward, done)
                state = next_state

            env.stats.invalid_action_cnt = agent.invalid_action_cnt
            env.stats.blank_actions = agent.blank_actions
            print('Grid Test: ', grid_size)
            print('---------- Evalutated %s runs ' % args.n_runs)
            if isinstance(agent, GPT3Agent):
                print('---------- All successes: ', env.stats.n_success)
                print('Average invalid actions: ', agent.invalid_action_cnt/ args.n_runs)
                print('Average blank actions: ', agent.blank_actions / args.n_runs)
            print('Average success rate: ', env.stats.n_success/args.n_runs)
            if env.stats.n_non_success > 0:
                print('---------- All fails: ', env.stats.n_non_success)
                print('Average die ratio: ', env.stats.n_died/env.stats.n_non_success)
                print('Average failure ratio: ', env.stats.n_failed/env.stats.n_non_success)
            else:
                print('Never failed or died.')
            print('Median episode length: ', np.median(env.stats.episode_lengths))
            print('Median optimal episode length: ', np.median(env.stats.optimal_episode_lengths))
            if len(env.stats.episode_lengths) > 2:
                print('Correlation: ', pearsonr(env.stats.episode_lengths, env.stats.optimal_episode_lengths))
                
            # save results
            if args.policy_type == PolicyType.Pretrained:
                agent_infos = agent.id.split(':expert-')
                agent_name = agent_infos[0] + '_' + agent_infos[-1]
            else:
                agent_name = agent.id
            env.stats.save(f'{agent_name}_grid{grid_size[0]}x{grid_size[1]}_descr{args.add_description}_shift{args.domain_shift}')
            env.stats.reset()
            
            sleep(10) # API cooldown


if __name__ == '__main__':
    args = parse_args()
    run(args)
