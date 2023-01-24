#!/bin/bash

# create api key
export OPENAI_API_KEY="<OPENAI_API_KEY>"

openai api models.delete -i $1
