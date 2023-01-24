#!/bin/bash


# # generate data with description and no domain shift
# echo "Generating data with description and no domain shift"
# PYTHONPATH=. python eval.py --n_runs 20 --policy_type pretrained --grid_sizes "[(8, 8), (11, 11), (13, 9), (15, 10), (20, 20)]" --model_id "models.log" --add_description

# # generate data with no description and no domain shift
# echo "Generating data with no description and no domain shift"
# PYTHONPATH=. python eval.py --n_runs 20 --policy_type pretrained --grid_sizes "[(8, 8), (11, 11), (13, 9), (15, 10), (20, 20)]" --model_id "models.log"

# add random baseline
echo "Generating data with random baseline"
PYTHONPATH=. python eval.py --n_runs 20 --policy_type random --grid_sizes "[(8, 8), (11, 11), (13, 9), (15, 10), (20, 20)]"
