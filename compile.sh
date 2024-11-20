#!/usr/bin/env bash

mkdir -p build
cd build
cmake ..
make
cd ..
./build/task1
