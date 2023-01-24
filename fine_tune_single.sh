#!/bin/bash

# create api key
export OPENAI_API_KEY="<OPENAI_API_KEY>"


DATASET=data/$1.jsonl
echo $DATASET
openai tools fine_tunes.prepare_data -f $DATASET
openai api fine_tunes.create -t $DATASET -m babbage --suffix $1
