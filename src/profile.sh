#!/bin/bash
if [ -z $1 ]; then
    echo "Please specify a file to profile."
    exit 1
  fi

python3 -m cProfile -o out.profile $1 quiet plot=False
snakeviz out.profile

