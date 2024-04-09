#!/bin/sh

clang++ implicit-runge-kutta.cpp -o out -DNDEBUG -Ofast --std=c++20 -mtune=native -march=native -mavx2
./out > data
python plot.py
