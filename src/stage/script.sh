#!/bin/bash

python main.py 2> ./errors.log | tee ./output.log
