# Tsetlin Machine: A Breath of Fresh Air in Machine Learning


> *“Speed is the most important feature.”*

*Fred Wilson*

This repository provides an alternative [Fuzzy-Pattern Tsetlin Machine](https://github.com/BooBSD/FuzzyPatternTM) implementation with zero external dependencies and blazingly fast performance.
It achieves over **32 million** MNIST predictions per second at 98% accuracy, with a throughput of **4 GB/s** on a desktop CPU.


Key features
------------

  - Up to **7× faster training** and **11× faster inference** compared to the original FPTM implementation, achieved through the use of bitwise operations, SIMD instructions, and a specialized memory layout.
  - Binary classifier.
  - Multi-class classifier.
  - Single-threaded and multi-threaded training and inference.
  - *BitSet* literal indexing to improve performance on very large, sparse binary vector inputs.
  - Model compilation to reduce memory usage and increase inference speed.
  - Save and load trained models for production deployment or continued training with modified hyperparameters.
  - Automatic selection of `UInt8` or `UInt16` Tsetlin Automata based on the number of TA states.
  - Automatic switching between binary and multi-class classification depending on the dataset.
  - Built-in benchmarking tool.


Introduction
------------

Here is a quick "Hello, World!" example of a typical use case.

Importing the necessary functions and MNIST dataset:

```julia
using MLDatasets: MNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize

x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
```

Booleanizing input data (2 bits per pixel):

```julia
x_train = [booleanize(x, 0, 0.5) for x in x_train]
x_test = [booleanize(x, 0, 0.5) for x in x_test]
```

There are some different hyperparameters compared to the [Vanilla Tsetlin Machine](https://github.com/cair/tmu).
The hyperparameter `L` limits the number of included literals in a clause.
`best_tms_size` is the number of the best TM models collected during the training process.
After training, you can save this ensemble of models to your drive or increase accuracy by using Binomial Combinatorial Merge with the `combine()` function.

```julia
EPOCHS = 1000
CLAUSES = 512
T = 16
S = 30
L = 12
best_tms_size = 500
```

Training the Tsetlin Machine over 1000 epochs and saving the best TM model to disk:

```julia
tm = TMClassifier{eltype(y_test)}(CLAUSES, T, S, L=L, states_num=256, include_limit=220)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, best_tms_compile=true, shuffle=true, batch=true)
save(tms[1][1], "/tmp/tm_best.tm")
```

Load the best Tsetlin Machine model and calculate the actual test accuracy:

```julia
tm = load("/tmp/tm_best.tm")
println(accuracy(predict(tm, x_test), y_test))
```

How to run examples
-------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32 mnist.jl` where `32` is the number of your logical CPU cores.

Benchmark
---------
The maximum MNIST inference speed achieved is **208 million** predictions per second (with a throughput of **19.1 GB/s**) in batch mode on a Ryzen 7950X3D desktop CPU, utilizing 32 threads.

Trained and optimized models can be found in `./examples/models/`.

How to run MNIST inference benchmark:

0. Please close all other programs such as web browsers, antivirus software, torrent clients, music players, etc.
1. Go to the examples directory: `cd ./examples`
2. Run `julia --project=. -O3 -t 32 mnist_benchmark_inference.jl` where `32` is the number of your logical CPU cores.
