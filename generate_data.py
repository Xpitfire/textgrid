import argparse
import os
import ast
from typing import Tuple, List
from agents.base import Agent, OptimalAgent, RandomAgent
from agents.gpt3 import GPT3Agent
from envs.textgrid import GridEnv
from utils import PolicyType


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='maximum sequence length (default: 50)')
    parser.add_argument('--trap_prob', type=float, default=0.05,
                        help='probability of traps')
    parser.add_argument('--obst_prob', type=float, default=0.2,
                        help='probability of obstacles (rocks or trees)')
    parser.add_argument('--tree_prob', type=float, default=0.00,
                        help='probability of trees')
    parser.add_argument('--policy_type', type=str, default='expert',
                        help='define policy type to collect data')
    parser.add_argument('--add_description', action=argparse.BooleanOptionalAction,
                        help='add description to starting state')
    parser.add_argument('--domain_shift', action=argparse.BooleanOptionalAction,
                        help='add domain shift to description')
    parser.add_argument('--model_id', type=str,
                        help='set the name of the model to use')
    parser.add_argument('--grid_sizes', type=lambda a: ast.literal_eval(a), nargs='+',
                        default="[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]",
                        help='list of grid sizes to generate data for')
    args = parser.parse_args()
    
    if isinstance(args.grid_sizes[0], list):
        args.grid_sizes = args.grid_sizes[0]
    
    return args


def generate_data(args):
    grids: List[Tuple[int, int]] = []
    episodes: List[str] = []
    
    info = 'shifted_' if args.domain_shift else ''
    info += 'descr_' if args.add_description else ''
    
    # generate data
    for grid_size in args.grid_sizes:
        env = GridEnv(grid_size,
                      obst_prob=args.obst_prob,
                      trap_prob=args.trap_prob,
                      tree_prob=args.tree_prob,
                      add_description=args.add_description)
        
        if args.policy_type == PolicyType.Expert:
            policy_name = 'expert'
            agent: Agent = OptimalAgent(env)
        elif args.policy_type == PolicyType.Random:
            policy_name = 'random'
            agent: Agent = RandomAgent(env)
        elif args.policy_type == PolicyType.Pretrained:
            policy_name = 'pretrained'
            agent: Agent = GPT3Agent(env, model_id=args.model_id)
        else:
            raise ValueError('Unsupported policy type')
        
        for _ in range(args.n_episodes):
            grids.append(grid_size)
            state = env.reset()
            episode = state
            done = False
            while not done:
                prep_state = env.prepare_state(state, grid_size, args.add_description, args.domain_shift)
                action = agent.action(prep_state)
                state, reward, done, _ = env.step(action)
                episode += state
            episodes.append(episode)
    
    os.makedirs('data', exist_ok=True)
    
    # write text minigrid
    policy_name = args.policy_type
    if args.policy_type == PolicyType.Pretrained:
        model_name = args.model_id.split(':')[0]
        policy_name = f'pretrained-{model_name}'
        if len(model_name) > 1:
            policy_name += '-ft'
    with open(f'data/{policy_name}_ex{args.n_episodes}_env{len(args.grid_sizes)}_{info}textminigrid_v1.jsonl', 'w') as f:
        for j, e in enumerate(episodes):
            states = e.split('>')
            for i in range(len(states)-1):
                state = states[i]
                if 'Game ended' not in state:
                    state += '>'
                
                prep_state = env.prepare_state(state, grids[j], args.add_description, args.domain_shift)
                prep_state = prep_state[prep_state.find('('):] if 'State 0' not in prep_state else prep_state
                
                completion = states[i+1]
                completion = completion[:completion.rfind(',')+1]
                
                line = '{"prompt": "%s", "completion": "%s \\n;;"}' % (prep_state, completion)
                line += '\n' 
                
                f.write(line)


if __name__ == '__main__':
    args = parse_args()
    generate_data(args)
