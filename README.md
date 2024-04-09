# Tsetlin Machine

The Tsetlin Machine library with zero external dependencies performs quite well.

<img src="https://raw.githubusercontent.com/BooBSD/Tsetlin.jl/main/raw/benchmark.png">

Key features
------------

  - Single-thread or multi-thread learning and inference.
  - Blazingly fast batch inference.
  - Compacting/shrinking TM models to save memory and increase inference speed.
  - Binomial combinatorial merge of trained models to achieve the best accuracy using two algorithms: merge and join.
  - Saving/loading trained models to/from disk.
  - Optimizing trained models by reordering included literals' indexes to maximize inference performance without batches.


Introduction
------------

Here is a quick "Hello, World!" example of a typical use case.

Importing the necessary functions and MNIST dataset:

```julia
using MLDatasets: MNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip

x_train, y_train = unzip([MNIST(split=:train)...])
x_test, y_test = unzip([MNIST(split=:test)...])
```

Booleanizing input data:

```julia
x_train = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.5 ? true : false for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.5 ? true : false for x in i];
])) for i in x_test]
```

There are some different hyperparameters compared to the [Vanilla Tsetlin Machine](https://github.com/cair/tmu).
The hyperparameter `R` is a float in the range of `0.0` to `1.0`.
To get the actual `R` from the Vanilla `S` parameter, use the following formula: `R = S / (S + 1)`.
The hyperparameter `L` limits the number of literals in a clause.
`best_tms_size` is the number of the best TM models collected during the training process.
After training, you can save this ensemble of models to your drive or increase accuracy by using Binomial Combinatorial Merge with the `combine()` function.

```julia
const EPOCHS = 1000
const CLAUSES = 2048
const T = 32
const R = 0.94
const L = 12
const best_tms_size = 500
```

Training the Tsetlin Machine over 1000 epochs and saving the best 500 compacted TM models to disk:

```julia
tm = TMClassifier(CLAUSES, T, R, L=L, states_num=256, include_limit=128)
_, tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, best_tms_compile=true, shuffle=true, batch=true)
save(tms, "/tmp/tms.tm")
```

Load the best Tsetlin Machine model and calculate the actual test accuracy:

```julia
tms = load("/tmp/tms.tm")
println(accuracy(predict(tms[1][2], x_test), y_test))
```

How to run examples
-------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32,1 --gcthreads=32,1 mnist_simple.jl` where `32` is the number of your logical CPU cores.

Benchmark
---------

Trained and optimized models can be found in `./examples/models/`.

How to run MNIST inference benchmark:

0. Please close all other programs such as web browsers, antivirus software, torrent trackers, music players, etc.
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32,1 mnist_benchmark_inference.jl` where `32` is the number of your logical CPU cores.



[![Build Status](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BooBSD/Tsetlin.jl/actions/workflows/CI.yml?query=branch%3Amain)
