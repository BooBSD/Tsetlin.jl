## How to Run Examples

- Ensure that you have the latest version of the [Julia language](https://julialang.org/downloads/) installed.
- Some examples require dataset preparation scripts written in [Python](https://www.python.org/downloads/). To install the necessary dependencies, run the following command:

```shell
pip install -r examples/requirements.txt
```

### MNIST Example

Run the MNIST training example:

```shell
julia -O3 -t auto examples/MNIST/mnist.jl
```

### Shakespeare character-level text generation

Below is an example of character-level text generation in the style of Shakespeare, implemented using FPTM with HDC hypervectors and Monte Carlo sparse context subsampling.

In the **first terminal window**, run the following command to train the model over multiple epochs:

```shell
julia -t auto examples/TEXT/train.jl
```

In the **second terminal window**, run the following command after each training epoch to observe how the quality of the generated text evolves from one epoch to the next:

```shell
julia examples/TEXT/sample.jl
```

After **400+** epochs, you should see output similar to the following:

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

If you want to train the text generator on your own corpus, remove `/tmp/tm_text.tm` and `/tmp/hvectors`, and replace `/tmp/input.txt` with your own text corpus.

Then rerun:

```shell
julia -t auto examples/TEXT/train.jl
```

### IMDb Example (1 clause per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=12800 --imdb-num-words=40000
```

Run the IMDb training and benchmarking example:

```shell
julia -O3 -t auto examples/IMDb/imdb_minimal.jl
```

### IMDb Example (200 clauses per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=65535 --imdb-num-words=70000
```

Run the IMDb training and benchmarking example:

```shell
julia -O3 -t auto examples/IMDb/imdb_optimal.jl
```

### Noisy Amazon Sales Example

Prepare the noisy Amazon Sales dataset:

```shell
python examples/AmazonSales/prepare_dataset.py --dataset_noise_ratio=0.005
```

Run the Noisy Amazon Sales training example:

```shell
julia -O3 -t auto examples/AmazonSales/amazon.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing

Run the Fashion-MNIST training example:

```shell
julia -O3 -t auto examples/FashionMNIST/fmnist_conv.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing and Data Augmentation

To achieve maximum test accuracy, apply data augmentation when preparing the Fashion-MNIST dataset:

```shell
julia -O3 -t auto examples/FashionMNIST/prepare_augmented_dataset.jl
```

Run the example that trains a large model on Fashion-MNIST:

```shell
julia -O3 -t auto examples/FashionMNIST/fmnist_conv_augmented.jl
```

### CIFAR-10 Example Using Convolutional Preprocessing

Prepare the CIFAR-10 dataset:

```shell
julia -O3 -t auto examples/CIFAR10/prepare_dataset.jl
```

Run the CIFAR-10 training example:

```shell
julia -O3 -t auto examples/CIFAR10/cifar10_conv.jl
```

### Noisy Parity Problem

Prepare the dataset:

```shell
python examples/NoisyParity/prepare_dataset.py
```

Run the Noisy Parity training example:

```shell
julia -O3 -t auto examples/NoisyParity/noisy_parity.jl
```
