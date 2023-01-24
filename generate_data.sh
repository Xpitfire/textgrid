#!/bin/bash


# generate data with no description and no domain shift
PYTHONPATH=. python generate_data.py --n_episodes 1 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 5 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 10 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 20 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 50 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 100 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"

PYTHONPATH=. python generate_data.py --n_episodes 250 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]"


# generate data with descriptions and no domain shift
PYTHONPATH=. python generate_data.py --n_episodes 1 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 5 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 10 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 20 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 50 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 100 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

PYTHONPATH=. python generate_data.py --n_episodes 250 --policy_type expert --grid_sizes "[(7, 7), (9, 9), (10, 10), (13, 13), (15, 15)]" --add_description

