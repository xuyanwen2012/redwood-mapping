#!/bin/bash

cmake -S . -B build/ -D CMAKE_BUILD_TYPE:STRING=Release
cmake --build build --target all --config Release 

