## How to Run Examples

- Ensure that you have the latest version of the [Julia](https://julialang.org/downloads/) language installed.
- Some examples require dataset preparation scripts written in [Python](https://www.python.org/downloads/). To install the necessary dependencies, run the following command:

```shell
pip install -r examples/requirements.txt
```
In *all* Julia examples, we use `-t 32`, which specifies the use of `32` logical CPU cores.
Please adjust this parameter to match the actual number of logical cores available on your machine.

### IMDb Example (1 clause per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=12800 --imdb-num-words=40000
```

Run the IMDb training and benchmarking example:

```shell
julia --project=. -O3 -t 32 examples/IMDb/imdb_minimal.jl
```

### IMDb Example (200 clauses per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=65535 --imdb-num-words=70000
```

Run the IMDb training and benchmarking example:

```shell
julia --project=. -O3 -t 32 examples/IMDb/imdb_optimal.jl
```

### Noisy Amazon Sales Example

Prepare the noisy Amazon Sales dataset:

```shell
python examples/AmazonSales/prepare_dataset.py --dataset_noise_ratio=0.005
```

Run the Noisy Amazon Sales training example:

```shell
julia --project=. -O3 -t 32 examples/AmazonSales/amazon.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing

Run the Fashion-MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing and Data Augmentation

To achieve maximum test accuracy, apply data augmentation when preparing the Fashion-MNIST dataset:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/prepare_augmented_dataset.jl
```

Run the example that trains a large model on Fashion-MNIST:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv_augmented.jl
```

### CIFAR-10 Example Using Convolutional Preprocessing

Prepare the CIFAR-10 dataset:

```shell
julia --project=. -O3 -t 32 examples/CIFAR10/prepare_dataset.jl
```

Run the CIFAR-10 training example:

```shell
julia --project=. -O3 -t 32 examples/CIFAR10/cifar10_conv.jl
```

### MNIST Example

Run the MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/MNIST/mnist.jl
```

To run the MNIST inference benchmark, please use the following command:

```shell
julia --project=. -O3 -t 32 examples/MNIST/mnist_benchmark_inference.jl
```
