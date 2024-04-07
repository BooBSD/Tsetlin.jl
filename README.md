# Tsetlin Machine

The Tsetlin Machine library with zero external dependencies performs quite well.

<img src="https://raw.githubusercontent.com/BooBSD/Tsetlin.jl/main/raw/benchmark.png">

Introduction
------------

Here is a quick "Hello, World!" example of a typical use case:

Importing the MNIST dataset and booleanizing input data:

```julia
using MLDatasets: MNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip

x_train, y_train = unzip([m for m in MNIST(split=:train)])
x_test, y_test = unzip([m for m in MNIST(split=:test)])

x_train = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.5 true else false end for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.5 true else false end for x in i];
])) for i in x_test]
```

Training the Tsetlin Machine over 1000 epochs and saving the best 500 compacted TM models to disk:

```julia
const EPOCHS = 1000
const CLAUSES = 2048
const T = 32
const R = 0.94
const L = 12
const best_tms_size = 500

tm = TMClassifier(CLAUSES, T, R, L=L, states_num=256, include_limit=128)
_, tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, best_tms_compile=true, shuffle=true, batch=true)
save(tms, "/tmp/tms.tm")
```

Load the best Tsetlin Machine model and calculate the actual test accuracy:

```julia
tms = load("/tmp/tms.tm")
println(accuracy(predict(tms[1], x_test), y_test))
```

How to run examples
-------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32,1 --gcthreads=32,1 mnist_simple.jl` where `32` is the number of your logical CPU cores.


[![Build Status](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml?query=branch%3Amain)
