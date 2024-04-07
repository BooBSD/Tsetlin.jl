# Tsetlin Machine

The Tsetlin Machine library with zero external dependencies performs quite well.

<img src="https://raw.githubusercontent.com/BooBSD/Tsetlin.jl/main/raw/benchmark.png">

How to run examples
-------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32,1 --gcthreads=32,1 mnist_simple.jl` where `32` is number of your logical CPU cores.


[![Build Status](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml?query=branch%3Amain)
