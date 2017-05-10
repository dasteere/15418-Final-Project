#!/bin/bash
nvcc cudaRiver.cu -g -O3 -m64 --gpu-architecture compute_35 -c -o objs/scan.o
gcc -g rank.c main.c objs/scan.o -std=c99 -L/usr/local/cuda/lib64/ -lcudart