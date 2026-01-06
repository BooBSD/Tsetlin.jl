# Tsetlin Machine: Fresh Thinking in ML


> *“Speed is the most important feature.”*

*Fred Wilson*

This repository provides an alternative [Fuzzy-Pattern Tsetlin Machine](https://github.com/BooBSD/FuzzyPatternTM) implementation with zero external dependencies and blazingly fast performance.
It achieves over **32 million** MNIST predictions per second at 98% accuracy, with a throughput of **4 GB/s** on a desktop CPU.


## Key Features

  - Up to **7× faster training** and **11× faster inference** compared to the original FPTM implementation, achieved through the use of bitwise operations, SIMD instructions, and a specialized memory layout.
  - Binary classifier.
  - Multi-class classifier.
  - Single-threaded and multi-threaded training and inference.
  - Specialized **BitSet index** for improved performance on very large, sparse binary vector inputs.
  - Model compilation to reduce memory usage and increase inference speed.
  - Save and load trained models for production deployment or continued training with modified hyperparameters.
  - Automatic selection of `UInt8` or `UInt16` Tsetlin Automata based on the number of TA states.
  - Automatic switching between binary and multi-class classification depending on the dataset.
  - Built-in benchmarking tool.


## Quick Start

> *Talk is cheap, show me the ~~code~~ some examples.*

First, install the [Julia language](https://julialang.org) by running the following command and following the installation instructions:

```shell
curl -fsSL https://install.julialang.org | sh
```

In the *first terminal window*, run the following command to train your model over multiple epochs:

```shell
julia -t auto examples/TEXT/text.jl
```

In the *second terminal window*, run the same command after each training epoch to observe how the text quality changes from one epoch to the next:

```shell
julia -t auto examples/TEXT/text.jl
```

After *200+* epochs, you should see output similar to the following:

```text
ROMEO:
The father's death,
And then I shall be so;
For I have done that was a queen,
That I may be so, my lord.

JULIET:
I would have should be so, for the prince,
And then I shall be so;
For the princely father with the princess,
And then I shall be the virtue of your soul,
Which your son,--

ESCALUS:
What, what should be particular me to death.

BUCKINGHAM:
God save the queen's proclaim'd:
Come, come, the Duke of York.

KING EDWARD IV:
So do I do not know the prince,
And then I shall be so, and such a part.

KING RICHARD III:
Shall I be some confess the state,
Which way the sun the prince's dead;
And then I will be so.
```

## Introduction

Here is a quick *"Hello, World!"* example of a typical use case with the Tsetlin Machine.

Importing the necessary functions and the MNIST dataset:

```julia
using MLDatasets: MNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize, compile, benchmark

x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
```

Booleanizing input data (2 bits per pixel):

```julia
x_train = [booleanize(x, 0, 0.5) for x in x_train]
x_test = [booleanize(x, 0, 0.5) for x in x_test]
```

### Hyperparameters
This implementation introduces some differences compared to the *Vanilla Tsetlin Machine*:

  - `L` — limits the number of included literals in a clause.
  - `LF` — new hyperparameter that sets the number of literal misses allowed per clause.

```julia
CLAUSES = 20   # Number of clauses per class
T       = 20   # Voting threshold
S       = 200  # Specificity
L       = 150  # Maximum literals per clause
LF      = 75   # Allowed failed literals per clause

EPOCHS  = 1000 # Number of training epochs
```

Train the model over 1000 epochs and save the compiled model to disk:

```julia
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=240)
train!(tm, x_train, y_train, x_test, y_test, EPOCHS, shuffle=true, index=false)
save(compile(tm), "/tmp/tm_last.tm")
```

Loading the compiled model and evaluating accuracy:

```julia
tm = load("/tmp/tm_last.tm")
println(accuracy(predict(tm, x_test), y_test))
```

Benchmarking the compiled model:

```julia
benchmark(tm, x_test, y_test, 1000 * 4, warmup=true, index=false)
```

## More Examples

This repository includes examples for *MNIST*, *Fashion-MNIST*, *CIFAR-10*, *AmazonSales*, *IMDb* sentiment analysis, and *Shakespeare* character-level **text generation**.

Instructions on how to run the examples can be found [here](https://github.com/BooBSD/Tsetlin.jl/tree/main/examples).
