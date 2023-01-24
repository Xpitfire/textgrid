#!/bin/bash

# create api key
export OPENAI_API_KEY="<OPENAI_API_KEY>"

openai api models.list > models.log
