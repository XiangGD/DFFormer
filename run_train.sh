#!/bin/bash
torchrun --nproc_per_node=3 \
         train.py