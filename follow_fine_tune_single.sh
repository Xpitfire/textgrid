#!/bin/bash

# create api key
export OPENAI_API_KEY="<OPENAI_API_KEY>"

openai api fine_tunes.follow -i $1
