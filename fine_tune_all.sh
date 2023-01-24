#!/bin/bash

# create api key
export OPENAI_API_KEY="<OPENAI_API_KEY>"

MODEL="ada"

echo "Training with $MODEL"

# with state descriptions

NAME="expert_ex1_env5_descr_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex5_env5_descr_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex10_env5_descr_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex20_env5_descr_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex50_env5_descr_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

# NAME="expert_ex100_env5_descr_textminigrid_v1"
# DATASET="data/$NAME.jsonl"
# echo $DATASET
# openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

# NAME="expert_ex250_env5_descr_textminigrid_v1"
# DATASET="data/$NAME.jsonl"
# echo $DATASET
# openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &


# without descriptions

NAME="expert_ex1_env5_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex5_env5_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex10_env5_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex20_env5_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

NAME="expert_ex50_env5_textminigrid_v1"
DATASET="data/$NAME.jsonl"
echo $DATASET
openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

# NAME="expert_ex100_env5_textminigrid_v1"
# DATASET="data/$NAME.jsonl"
# echo $DATASET
# openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &

# NAME="expert_ex250_env5_textminigrid_v1"
# DATASET="data/$NAME.jsonl"
# echo $DATASET
# openai api fine_tunes.create -t $DATASET -m $MODEL --suffix $NAME > results/out_$NAME.log &
