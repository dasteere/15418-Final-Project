#!/bin/bash
nvcc cudaRiver.cu -g -O0 -m64 --gpu-architecture compute_35 -c -o objs/scan.o
gcc -Wall -O0 -g rank.c main.c output_utils.c cpu-sequential.c objs/scan.o -std=c99 -L/usr/local/cuda/lib64/ -lcudart
