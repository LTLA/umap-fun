#!/bin/bash

emcmake cmake -S . -B build
cd build && emmake make
