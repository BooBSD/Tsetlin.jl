include("../../src/Tsetlin.jl")
include("../../src/utils/fastconv.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Base.Threads
using Serialization
using Statistics
using MLDatasets: CIFAR10
using .Tsetlin: InputVector, InputBatch, unzip

x_train, y_train = unzip([CIFAR10(:train)...])
x_test, y_test = unzip([CIFAR10(:test)...])

const x_trainR = [x[:, :, 1] for x in x_train]
const x_trainG = [x[:, :, 2] for x in x_train]
const x_trainB = [x[:, :, 3] for x in x_train]
const x_testR = [x[:, :, 1] for x in x_test]
const x_testG = [x[:, :, 2] for x in x_test]
const x_testB = [x[:, :, 3] for x in x_test]

print("Preparing input data... ")

# Convolution kernels
const Kx3 = [-1 0 1; -2 0 2; -1 0 1] * one(Float32)
# const Kx5 = [-2 -1 0 1 2; -3 -2 0 2 3; -4 -3 0 3 4; -3 -2 0 2 3; -2 -1 0 1 2] * one(Float32)
const Kx7 = [-3 -2 -1 0 1 2 3; -4 -3 -2 0 2 3 4; -5 -4 -3 0 3 4 5; -6 -5 -4 0 4 5 6; -5 -4 -3 0 3 4 5; -4 -3 -2 0 2 3 4; -3 -2 -1 0 1 2 3] * one(Float32)

# const Kx3 = [0 1 2; -1 0 1; -2 -1 0] * one(Float32)
const Kx5 = [0 1 2 3 4; -1 0 2 3 3; -2 -2 0 2 2; -3 -3 -2 0 1; -4 -3 -2 -1 0] * one(Float32)
# const Kx7 = [0 1 2 3 4 5 6; -1 0 2 3 4 5 5; -2 -2 0 3 4 4 4; -3 -3 -3 0 3 3 3; -4 -4 -4 -3 0 2 2; -5 -5 -4 -3 -2 0 1; -6 -5 -4 -3 -2 -1 0] * one(Float32)

const Kx9 = [-1 -1 -1; 2 2 2; -1 -1 -1] * one(Float32)

const Ky3 = rotl90(Kx3)
const Ky5 = rotl90(Kx5)
const Ky7 = rotl90(Kx7)
const Ky9 = rotl90(Kx9)

const Kp3 = 1  # Padding 1
const Kp5 = 2  # Padding 2
const Kp7 = 3  # Padding 3
const Kp9 = 1  # Padding 1

const x_train_conv_orient_x3R = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainR]
const x_train_conv_orient_y3R = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainR]

const x_test_conv_orient_x3R = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testR]
const x_test_conv_orient_y3R = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testR]

const x_train_conv_orient_x5R = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainR]
const x_train_conv_orient_y5R = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainR]

const x_test_conv_orient_x5R = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testR]
const x_test_conv_orient_y5R = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testR]

const x_train_conv_orient_x7R = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainR]
const x_train_conv_orient_y7R = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainR]

const x_test_conv_orient_x7R = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testR]
const x_test_conv_orient_y7R = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testR]

const x_train_conv_orient_x9R = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainR]
const x_train_conv_orient_y9R = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainR]

const x_test_conv_orient_x9R = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testR]
const x_test_conv_orient_y9R = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testR]


const x_train_conv_orient_x3G = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainG]
const x_train_conv_orient_y3G = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainG]

const x_test_conv_orient_x3G = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testG]
const x_test_conv_orient_y3G = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testG]

const x_train_conv_orient_x5G = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainG]
const x_train_conv_orient_y5G = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainG]

const x_test_conv_orient_x5G = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testG]
const x_test_conv_orient_y5G = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testG]

const x_train_conv_orient_x7G = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainG]
const x_train_conv_orient_y7G = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainG]

const x_test_conv_orient_x7G = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testG]
const x_test_conv_orient_y7G = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testG]

const x_train_conv_orient_x9G = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainG]
const x_train_conv_orient_y9G = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainG]

const x_test_conv_orient_x9G = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testG]
const x_test_conv_orient_y9G = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testG]


const x_train_conv_orient_x3B = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainB]
const x_train_conv_orient_y3B = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_trainB]

const x_test_conv_orient_x3B = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testB]
const x_test_conv_orient_y3B = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_testB]

const x_train_conv_orient_x5B = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainB]
const x_train_conv_orient_y5B = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_trainB]

const x_test_conv_orient_x5B = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testB]
const x_test_conv_orient_y5B = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_testB]

const x_train_conv_orient_x7B = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainB]
const x_train_conv_orient_y7B = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_trainB]

const x_test_conv_orient_x7B = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testB]
const x_test_conv_orient_y7B = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_testB]

const x_train_conv_orient_x9B = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainB]
const x_train_conv_orient_y9B = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_trainB]

const x_test_conv_orient_x9B = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testB]
const x_test_conv_orient_y9B = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_testB]


const train_hist_dataR = vec(vcat(x_trainR...))
const train_hist_dataG = vec(vcat(x_trainG...))
const train_hist_dataB = vec(vcat(x_trainB...))

const x3_hist_dataR = vec(vcat(x_train_conv_orient_x3R...))
const y3_hist_dataR = vec(vcat(x_train_conv_orient_y3R...))
const x5_hist_dataR = vec(vcat(x_train_conv_orient_x5R...))
const y5_hist_dataR = vec(vcat(x_train_conv_orient_y5R...))
const x7_hist_dataR = vec(vcat(x_train_conv_orient_x7R...))
const y7_hist_dataR = vec(vcat(x_train_conv_orient_y7R...))
const x9_hist_dataR = vec(vcat(x_train_conv_orient_x9R...))
const y9_hist_dataR = vec(vcat(x_train_conv_orient_y9R...))

const x3_hist_dataG = vec(vcat(x_train_conv_orient_x3G...))
const y3_hist_dataG = vec(vcat(x_train_conv_orient_y3G...))
const x5_hist_dataG = vec(vcat(x_train_conv_orient_x5G...))
const y5_hist_dataG = vec(vcat(x_train_conv_orient_y5G...))
const x7_hist_dataG = vec(vcat(x_train_conv_orient_x7G...))
const y7_hist_dataG = vec(vcat(x_train_conv_orient_y7G...))
const x9_hist_dataG = vec(vcat(x_train_conv_orient_x9G...))
const y9_hist_dataG = vec(vcat(x_train_conv_orient_y9G...))

const x3_hist_dataB = vec(vcat(x_train_conv_orient_x3B...))
const y3_hist_dataB = vec(vcat(x_train_conv_orient_y3B...))
const x5_hist_dataB = vec(vcat(x_train_conv_orient_x5B...))
const y5_hist_dataB = vec(vcat(x_train_conv_orient_y5B...))
const x7_hist_dataB = vec(vcat(x_train_conv_orient_x7B...))
const y7_hist_dataB = vec(vcat(x_train_conv_orient_y7B...))
const x9_hist_dataB = vec(vcat(x_train_conv_orient_x9B...))
const y9_hist_dataB = vec(vcat(x_train_conv_orient_y9B...))


const raw_hist_25R::Float64 = quantile([x for x in train_hist_dataR if x > 0], 0.25)
const raw_hist_50R::Float64 = quantile([x for x in train_hist_dataR if x > 0], 0.50)
const raw_hist_75R::Float64= quantile([x for x in train_hist_dataR if x > 0], 0.75)

const x3_hist_pos_25R::Float64 = quantile([x for x in x3_hist_dataR if x > 0], 0.25)
const x3_hist_pos_34R::Float64 = quantile([x for x in x3_hist_dataR if x > 0], 0.34)
const x3_hist_pos_50R::Float64 = quantile([x for x in x3_hist_dataR if x > 0], 0.50)
const x3_hist_pos_75R::Float64 = quantile([x for x in x3_hist_dataR if x > 0], 0.75)
const x3_hist_neg_25R::Float64 = quantile([x for x in x3_hist_dataR if x < 0], 1 - 0.25)
const x3_hist_neg_34R::Float64 = quantile([x for x in x3_hist_dataR if x < 0], 1 - 0.34)
const x3_hist_neg_50R::Float64 = quantile([x for x in x3_hist_dataR if x < 0], 1 - 0.50)
const x3_hist_neg_75R::Float64 = quantile([x for x in x3_hist_dataR if x < 0], 1 - 0.75)

const y3_hist_pos_25R::Float64 = quantile([x for x in y3_hist_dataR if x > 0], 0.25)
const y3_hist_pos_34R::Float64 = quantile([x for x in y3_hist_dataR if x > 0], 0.34)
const y3_hist_pos_50R::Float64 = quantile([x for x in y3_hist_dataR if x > 0], 0.50)
const y3_hist_pos_75R::Float64 = quantile([x for x in y3_hist_dataR if x > 0], 0.75)
const y3_hist_neg_25R::Float64 = quantile([x for x in y3_hist_dataR if x < 0], 1 - 0.25)
const y3_hist_neg_34R::Float64 = quantile([x for x in y3_hist_dataR if x < 0], 1 - 0.34)
const y3_hist_neg_50R::Float64 = quantile([x for x in y3_hist_dataR if x < 0], 1 - 0.50)
const y3_hist_neg_75R::Float64 = quantile([x for x in y3_hist_dataR if x < 0], 1 - 0.75)

const x5_hist_pos_25R::Float64 = quantile([x for x in x5_hist_dataR if x > 0], 0.25)
const x5_hist_pos_34R::Float64 = quantile([x for x in x5_hist_dataR if x > 0], 0.34)
const x5_hist_pos_50R::Float64 = quantile([x for x in x5_hist_dataR if x > 0], 0.50)
const x5_hist_pos_75R::Float64 = quantile([x for x in x5_hist_dataR if x > 0], 0.75)
const x5_hist_neg_25R::Float64 = quantile([x for x in x5_hist_dataR if x < 0], 1 - 0.25)
const x5_hist_neg_34R::Float64 = quantile([x for x in x5_hist_dataR if x < 0], 1 - 0.34)
const x5_hist_neg_50R::Float64 = quantile([x for x in x5_hist_dataR if x < 0], 1 - 0.50)
const x5_hist_neg_75R::Float64 = quantile([x for x in x5_hist_dataR if x < 0], 1 - 0.75)

const y5_hist_pos_25R::Float64 = quantile([x for x in y5_hist_dataR if x > 0], 0.25)
const y5_hist_pos_34R::Float64 = quantile([x for x in y5_hist_dataR if x > 0], 0.34)
const y5_hist_pos_50R::Float64 = quantile([x for x in y5_hist_dataR if x > 0], 0.50)
const y5_hist_pos_75R::Float64 = quantile([x for x in y5_hist_dataR if x > 0], 0.75)
const y5_hist_neg_25R::Float64 = quantile([x for x in y5_hist_dataR if x < 0], 1 - 0.25)
const y5_hist_neg_34R::Float64 = quantile([x for x in y5_hist_dataR if x < 0], 1 - 0.34)
const y5_hist_neg_50R::Float64 = quantile([x for x in y5_hist_dataR if x < 0], 1 - 0.50)
const y5_hist_neg_75R::Float64 = quantile([x for x in y5_hist_dataR if x < 0], 1 - 0.75)

const x7_hist_pos_25R::Float64 = quantile([x for x in x7_hist_dataR if x > 0], 0.25)
const x7_hist_pos_34R::Float64 = quantile([x for x in x7_hist_dataR if x > 0], 0.34)
const x7_hist_pos_50R::Float64 = quantile([x for x in x7_hist_dataR if x > 0], 0.50)
const x7_hist_pos_75R::Float64 = quantile([x for x in x7_hist_dataR if x > 0], 0.75)
const x7_hist_neg_25R::Float64 = quantile([x for x in x7_hist_dataR if x < 0], 1 - 0.25)
const x7_hist_neg_34R::Float64 = quantile([x for x in x7_hist_dataR if x < 0], 1 - 0.34)
const x7_hist_neg_50R::Float64 = quantile([x for x in x7_hist_dataR if x < 0], 1 - 0.50)
const x7_hist_neg_75R::Float64 = quantile([x for x in x7_hist_dataR if x < 0], 1 - 0.75)

const y7_hist_pos_25R::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.25)
const y7_hist_pos_34R::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.34)
const y7_hist_pos_50R::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.50)
const y7_hist_pos_75R::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.75)
const y7_hist_neg_25R::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.25)
const y7_hist_neg_34R::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.34)
const y7_hist_neg_50R::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.50)
const y7_hist_neg_75R::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.75)

const x9_hist_pos_25R::Float64 = quantile([x for x in x9_hist_dataR if x > 0], 0.25)
const x9_hist_pos_34R::Float64 = quantile([x for x in x9_hist_dataR if x > 0], 0.34)
const x9_hist_pos_50R::Float64 = quantile([x for x in x9_hist_dataR if x > 0], 0.50)
const x9_hist_pos_75R::Float64 = quantile([x for x in x9_hist_dataR if x > 0], 0.75)
const x9_hist_neg_25R::Float64 = quantile([x for x in x9_hist_dataR if x < 0], 1 - 0.25)
const x9_hist_neg_34R::Float64 = quantile([x for x in x9_hist_dataR if x < 0], 1 - 0.34)
const x9_hist_neg_50R::Float64 = quantile([x for x in x9_hist_dataR if x < 0], 1 - 0.50)
const x9_hist_neg_75R::Float64 = quantile([x for x in x9_hist_dataR if x < 0], 1 - 0.75)

const y9_hist_pos_25R::Float64 = quantile([x for x in y9_hist_dataR if x > 0], 0.25)
const y9_hist_pos_34R::Float64 = quantile([x for x in y9_hist_dataR if x > 0], 0.34)
const y9_hist_pos_50R::Float64 = quantile([x for x in y9_hist_dataR if x > 0], 0.50)
const y9_hist_pos_75R::Float64 = quantile([x for x in y9_hist_dataR if x > 0], 0.75)
const y9_hist_neg_25R::Float64 = quantile([x for x in y9_hist_dataR if x < 0], 1 - 0.25)
const y9_hist_neg_34R::Float64 = quantile([x for x in y9_hist_dataR if x < 0], 1 - 0.34)
const y9_hist_neg_50R::Float64 = quantile([x for x in y9_hist_dataR if x < 0], 1 - 0.50)
const y9_hist_neg_75R::Float64 = quantile([x for x in y9_hist_dataR if x < 0], 1 - 0.75)


const raw_hist_25G::Float64 = quantile([x for x in train_hist_dataG if x > 0], 0.25)
const raw_hist_50G::Float64 = quantile([x for x in train_hist_dataG if x > 0], 0.50)
const raw_hist_75G::Float64= quantile([x for x in train_hist_dataG if x > 0], 0.75)

const x3_hist_pos_25G::Float64 = quantile([x for x in x3_hist_dataG if x > 0], 0.25)
const x3_hist_pos_34G::Float64 = quantile([x for x in x3_hist_dataG if x > 0], 0.34)
const x3_hist_pos_50G::Float64 = quantile([x for x in x3_hist_dataG if x > 0], 0.50)
const x3_hist_pos_75G::Float64 = quantile([x for x in x3_hist_dataG if x > 0], 0.75)
const x3_hist_neg_25G::Float64 = quantile([x for x in x3_hist_dataG if x < 0], 1 - 0.25)
const x3_hist_neg_34G::Float64 = quantile([x for x in x3_hist_dataG if x < 0], 1 - 0.34)
const x3_hist_neg_50G::Float64 = quantile([x for x in x3_hist_dataG if x < 0], 1 - 0.50)
const x3_hist_neg_75G::Float64 = quantile([x for x in x3_hist_dataG if x < 0], 1 - 0.75)

const y3_hist_pos_25G::Float64 = quantile([x for x in y3_hist_dataG if x > 0], 0.25)
const y3_hist_pos_34G::Float64 = quantile([x for x in y3_hist_dataG if x > 0], 0.34)
const y3_hist_pos_50G::Float64 = quantile([x for x in y3_hist_dataG if x > 0], 0.50)
const y3_hist_pos_75G::Float64 = quantile([x for x in y3_hist_dataG if x > 0], 0.75)
const y3_hist_neg_25G::Float64 = quantile([x for x in y3_hist_dataG if x < 0], 1 - 0.25)
const y3_hist_neg_34G::Float64 = quantile([x for x in y3_hist_dataG if x < 0], 1 - 0.34)
const y3_hist_neg_50G::Float64 = quantile([x for x in y3_hist_dataG if x < 0], 1 - 0.50)
const y3_hist_neg_75G::Float64 = quantile([x for x in y3_hist_dataG if x < 0], 1 - 0.75)

const x5_hist_pos_25G::Float64 = quantile([x for x in x5_hist_dataG if x > 0], 0.25)
const x5_hist_pos_34G::Float64 = quantile([x for x in x5_hist_dataG if x > 0], 0.34)
const x5_hist_pos_50G::Float64 = quantile([x for x in x5_hist_dataG if x > 0], 0.50)
const x5_hist_pos_75G::Float64 = quantile([x for x in x5_hist_dataG if x > 0], 0.75)
const x5_hist_neg_25G::Float64 = quantile([x for x in x5_hist_dataG if x < 0], 1 - 0.25)
const x5_hist_neg_34G::Float64 = quantile([x for x in x5_hist_dataG if x < 0], 1 - 0.34)
const x5_hist_neg_50G::Float64 = quantile([x for x in x5_hist_dataG if x < 0], 1 - 0.50)
const x5_hist_neg_75G::Float64 = quantile([x for x in x5_hist_dataG if x < 0], 1 - 0.75)

const y5_hist_pos_25G::Float64 = quantile([x for x in y5_hist_dataG if x > 0], 0.25)
const y5_hist_pos_34G::Float64 = quantile([x for x in y5_hist_dataG if x > 0], 0.34)
const y5_hist_pos_50G::Float64 = quantile([x for x in y5_hist_dataG if x > 0], 0.50)
const y5_hist_pos_75G::Float64 = quantile([x for x in y5_hist_dataG if x > 0], 0.75)
const y5_hist_neg_25G::Float64 = quantile([x for x in y5_hist_dataG if x < 0], 1 - 0.25)
const y5_hist_neg_34G::Float64 = quantile([x for x in y5_hist_dataG if x < 0], 1 - 0.34)
const y5_hist_neg_50G::Float64 = quantile([x for x in y5_hist_dataG if x < 0], 1 - 0.50)
const y5_hist_neg_75G::Float64 = quantile([x for x in y5_hist_dataG if x < 0], 1 - 0.75)

const x7_hist_pos_25G::Float64 = quantile([x for x in x7_hist_dataG if x > 0], 0.25)
const x7_hist_pos_34G::Float64 = quantile([x for x in x7_hist_dataG if x > 0], 0.34)
const x7_hist_pos_50G::Float64 = quantile([x for x in x7_hist_dataG if x > 0], 0.50)
const x7_hist_pos_75G::Float64 = quantile([x for x in x7_hist_dataG if x > 0], 0.75)
const x7_hist_neg_25G::Float64 = quantile([x for x in x7_hist_dataG if x < 0], 1 - 0.25)
const x7_hist_neg_34G::Float64 = quantile([x for x in x7_hist_dataG if x < 0], 1 - 0.34)
const x7_hist_neg_50G::Float64 = quantile([x for x in x7_hist_dataG if x < 0], 1 - 0.50)
const x7_hist_neg_75G::Float64 = quantile([x for x in x7_hist_dataG if x < 0], 1 - 0.75)

const y7_hist_pos_25G::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.25)
const y7_hist_pos_34G::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.34)
const y7_hist_pos_50G::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.50)
const y7_hist_pos_75G::Float64 = quantile([x for x in y7_hist_dataR if x > 0], 0.75)
const y7_hist_neg_25G::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.25)
const y7_hist_neg_34G::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.34)
const y7_hist_neg_50G::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.50)
const y7_hist_neg_75G::Float64 = quantile([x for x in y7_hist_dataR if x < 0], 1 - 0.75)

const x9_hist_pos_25G::Float64 = quantile([x for x in x9_hist_dataG if x > 0], 0.25)
const x9_hist_pos_34G::Float64 = quantile([x for x in x9_hist_dataG if x > 0], 0.34)
const x9_hist_pos_50G::Float64 = quantile([x for x in x9_hist_dataG if x > 0], 0.50)
const x9_hist_pos_75G::Float64 = quantile([x for x in x9_hist_dataG if x > 0], 0.75)
const x9_hist_neg_25G::Float64 = quantile([x for x in x9_hist_dataG if x < 0], 1 - 0.25)
const x9_hist_neg_34G::Float64 = quantile([x for x in x9_hist_dataG if x < 0], 1 - 0.34)
const x9_hist_neg_50G::Float64 = quantile([x for x in x9_hist_dataG if x < 0], 1 - 0.50)
const x9_hist_neg_75G::Float64 = quantile([x for x in x9_hist_dataG if x < 0], 1 - 0.75)

const y9_hist_pos_25G::Float64 = quantile([x for x in y9_hist_dataG if x > 0], 0.25)
const y9_hist_pos_34G::Float64 = quantile([x for x in y9_hist_dataG if x > 0], 0.34)
const y9_hist_pos_50G::Float64 = quantile([x for x in y9_hist_dataG if x > 0], 0.50)
const y9_hist_pos_75G::Float64 = quantile([x for x in y9_hist_dataG if x > 0], 0.75)
const y9_hist_neg_25G::Float64 = quantile([x for x in y9_hist_dataG if x < 0], 1 - 0.25)
const y9_hist_neg_34G::Float64 = quantile([x for x in y9_hist_dataG if x < 0], 1 - 0.34)
const y9_hist_neg_50G::Float64 = quantile([x for x in y9_hist_dataG if x < 0], 1 - 0.50)
const y9_hist_neg_75G::Float64 = quantile([x for x in y9_hist_dataG if x < 0], 1 - 0.75)


const raw_hist_25B::Float64 = quantile([x for x in train_hist_dataB if x > 0], 0.25)
const raw_hist_50B::Float64 = quantile([x for x in train_hist_dataB if x > 0], 0.50)
const raw_hist_75B::Float64= quantile([x for x in train_hist_dataB if x > 0], 0.75)

const x3_hist_pos_25B::Float64 = quantile([x for x in x3_hist_dataB if x > 0], 0.25)
const x3_hist_pos_34B::Float64 = quantile([x for x in x3_hist_dataB if x > 0], 0.34)
const x3_hist_pos_50B::Float64 = quantile([x for x in x3_hist_dataB if x > 0], 0.50)
const x3_hist_pos_75B::Float64 = quantile([x for x in x3_hist_dataB if x > 0], 0.75)
const x3_hist_neg_25B::Float64 = quantile([x for x in x3_hist_dataB if x < 0], 1 - 0.25)
const x3_hist_neg_34B::Float64 = quantile([x for x in x3_hist_dataB if x < 0], 1 - 0.34)
const x3_hist_neg_50B::Float64 = quantile([x for x in x3_hist_dataB if x < 0], 1 - 0.50)
const x3_hist_neg_75B::Float64 = quantile([x for x in x3_hist_dataB if x < 0], 1 - 0.75)

const y3_hist_pos_25B::Float64 = quantile([x for x in y3_hist_dataB if x > 0], 0.25)
const y3_hist_pos_34B::Float64 = quantile([x for x in y3_hist_dataB if x > 0], 0.34)
const y3_hist_pos_50B::Float64 = quantile([x for x in y3_hist_dataB if x > 0], 0.50)
const y3_hist_pos_75B::Float64 = quantile([x for x in y3_hist_dataB if x > 0], 0.75)
const y3_hist_neg_25B::Float64 = quantile([x for x in y3_hist_dataB if x < 0], 1 - 0.25)
const y3_hist_neg_34B::Float64 = quantile([x for x in y3_hist_dataB if x < 0], 1 - 0.34)
const y3_hist_neg_50B::Float64 = quantile([x for x in y3_hist_dataB if x < 0], 1 - 0.50)
const y3_hist_neg_75B::Float64 = quantile([x for x in y3_hist_dataB if x < 0], 1 - 0.75)

const x5_hist_pos_25B::Float64 = quantile([x for x in x5_hist_dataB if x > 0], 0.25)
const x5_hist_pos_34B::Float64 = quantile([x for x in x5_hist_dataB if x > 0], 0.34)
const x5_hist_pos_50B::Float64 = quantile([x for x in x5_hist_dataB if x > 0], 0.50)
const x5_hist_pos_75B::Float64 = quantile([x for x in x5_hist_dataB if x > 0], 0.75)
const x5_hist_neg_25B::Float64 = quantile([x for x in x5_hist_dataB if x < 0], 1 - 0.25)
const x5_hist_neg_34B::Float64 = quantile([x for x in x5_hist_dataB if x < 0], 1 - 0.34)
const x5_hist_neg_50B::Float64 = quantile([x for x in x5_hist_dataB if x < 0], 1 - 0.50)
const x5_hist_neg_75B::Float64 = quantile([x for x in x5_hist_dataB if x < 0], 1 - 0.75)

const y5_hist_pos_25B::Float64 = quantile([x for x in y5_hist_dataB if x > 0], 0.25)
const y5_hist_pos_34B::Float64 = quantile([x for x in y5_hist_dataB if x > 0], 0.34)
const y5_hist_pos_50B::Float64 = quantile([x for x in y5_hist_dataB if x > 0], 0.50)
const y5_hist_pos_75B::Float64 = quantile([x for x in y5_hist_dataB if x > 0], 0.75)
const y5_hist_neg_25B::Float64 = quantile([x for x in y5_hist_dataB if x < 0], 1 - 0.25)
const y5_hist_neg_34B::Float64 = quantile([x for x in y5_hist_dataB if x < 0], 1 - 0.34)
const y5_hist_neg_50B::Float64 = quantile([x for x in y5_hist_dataB if x < 0], 1 - 0.50)
const y5_hist_neg_75B::Float64 = quantile([x for x in y5_hist_dataB if x < 0], 1 - 0.75)

const x7_hist_pos_25B::Float64 = quantile([x for x in x7_hist_dataB if x > 0], 0.25)
const x7_hist_pos_34B::Float64 = quantile([x for x in x7_hist_dataB if x > 0], 0.34)
const x7_hist_pos_50B::Float64 = quantile([x for x in x7_hist_dataB if x > 0], 0.50)
const x7_hist_pos_75B::Float64 = quantile([x for x in x7_hist_dataB if x > 0], 0.75)
const x7_hist_neg_25B::Float64 = quantile([x for x in x7_hist_dataB if x < 0], 1 - 0.25)
const x7_hist_neg_34B::Float64 = quantile([x for x in x7_hist_dataB if x < 0], 1 - 0.34)
const x7_hist_neg_50B::Float64 = quantile([x for x in x7_hist_dataB if x < 0], 1 - 0.50)
const x7_hist_neg_75B::Float64 = quantile([x for x in x7_hist_dataB if x < 0], 1 - 0.75)

const y7_hist_pos_25B::Float64 = quantile([x for x in y7_hist_dataB if x > 0], 0.25)
const y7_hist_pos_34B::Float64 = quantile([x for x in y7_hist_dataB if x > 0], 0.34)
const y7_hist_pos_50B::Float64 = quantile([x for x in y7_hist_dataB if x > 0], 0.50)
const y7_hist_pos_75B::Float64 = quantile([x for x in y7_hist_dataB if x > 0], 0.75)
const y7_hist_neg_25B::Float64 = quantile([x for x in y7_hist_dataB if x < 0], 1 - 0.25)
const y7_hist_neg_34B::Float64 = quantile([x for x in y7_hist_dataB if x < 0], 1 - 0.34)
const y7_hist_neg_50B::Float64 = quantile([x for x in y7_hist_dataB if x < 0], 1 - 0.50)
const y7_hist_neg_75B::Float64 = quantile([x for x in y7_hist_dataB if x < 0], 1 - 0.75)

const x9_hist_pos_25B::Float64 = quantile([x for x in x9_hist_dataB if x > 0], 0.25)
const x9_hist_pos_34B::Float64 = quantile([x for x in x9_hist_dataB if x > 0], 0.34)
const x9_hist_pos_50B::Float64 = quantile([x for x in x9_hist_dataB if x > 0], 0.50)
const x9_hist_pos_75B::Float64 = quantile([x for x in x9_hist_dataB if x > 0], 0.75)
const x9_hist_neg_25B::Float64 = quantile([x for x in x9_hist_dataB if x < 0], 1 - 0.25)
const x9_hist_neg_34B::Float64 = quantile([x for x in x9_hist_dataB if x < 0], 1 - 0.34)
const x9_hist_neg_50B::Float64 = quantile([x for x in x9_hist_dataB if x < 0], 1 - 0.50)
const x9_hist_neg_75B::Float64 = quantile([x for x in x9_hist_dataB if x < 0], 1 - 0.75)

const y9_hist_pos_25B::Float64 = quantile([x for x in y9_hist_dataB if x > 0], 0.25)
const y9_hist_pos_34B::Float64 = quantile([x for x in y9_hist_dataB if x > 0], 0.34)
const y9_hist_pos_50B::Float64 = quantile([x for x in y9_hist_dataB if x > 0], 0.50)
const y9_hist_pos_75B::Float64 = quantile([x for x in y9_hist_dataB if x > 0], 0.75)
const y9_hist_neg_25B::Float64 = quantile([x for x in y9_hist_dataB if x < 0], 1 - 0.25)
const y9_hist_neg_34B::Float64 = quantile([x for x in y9_hist_dataB if x < 0], 1 - 0.34)
const y9_hist_neg_50B::Float64 = quantile([x for x in y9_hist_dataB if x < 0], 1 - 0.50)
const y9_hist_neg_75B::Float64 = quantile([x for x in y9_hist_dataB if x < 0], 1 - 0.75)


# Booleanization
function bools(rawR, x3R, y3R, x5R, y5R, x7R, y7R, x9R, y9R, rawG, x3G, y3G, x5G, y5G, x7G, y7G, x9G, y9G, rawB, x3B, y3B, x5B, y5B, x7B, y7B, x9B, y9B)::BitVector
    return BitVector(vec([
        # Raw pixels
        [x > 0 for x in rawR];
        [x > raw_hist_25R for x in rawR];
        [x > raw_hist_50R for x in rawR];
        [x > raw_hist_75R for x in rawR];

        # 3x3 convolution results
        [x > 0 for x in x3R];
        [x > x3_hist_pos_25R for x in x3R];
        [x > x3_hist_pos_34R for x in x3R];
        [x > x3_hist_pos_50R for x in x3R];
        [x > x3_hist_pos_75R for x in x3R];
        [x < x3_hist_neg_25R for x in x3R];
        [x < x3_hist_neg_34R for x in x3R];
        [x < x3_hist_neg_50R for x in x3R];
        [x < x3_hist_neg_75R for x in x3R];

        [x > 0 for x in y3R];
        [x > y3_hist_pos_25R for x in y3R];
        [x > y3_hist_pos_34R for x in y3R];
        [x > y3_hist_pos_50R for x in y3R];
        [x > y3_hist_pos_75R for x in y3R];
        [x < y3_hist_neg_25R for x in y3R];
        [x < y3_hist_neg_34R for x in y3R];
        [x < y3_hist_neg_50R for x in y3R];
        [x < y3_hist_neg_75R for x in y3R];

        # 5x5 convolution results
        [x > 0 for x in x5R];
        [x > x5_hist_pos_25R for x in x5R];
        [x > x5_hist_pos_34R for x in x5R];
        [x > x5_hist_pos_50R for x in x5R];
        [x > x5_hist_pos_75R for x in x5R];
        [x < x5_hist_neg_25R for x in x5R];
        [x < x5_hist_neg_34R for x in x5R];
        [x < x5_hist_neg_50R for x in x5R];
        [x < x5_hist_neg_75R for x in x5R];

        [x > 0 for x in y5R];
        [x > y5_hist_pos_25R for x in y5R];
        [x > y5_hist_pos_34R for x in y5R];
        [x > y5_hist_pos_50R for x in y5R];
        [x > y5_hist_pos_75R for x in y5R];
        [x < y5_hist_neg_25R for x in y5R];
        [x < y5_hist_neg_34R for x in y5R];
        [x < y5_hist_neg_50R for x in y5R];
        [x < y5_hist_neg_75R for x in y5R];

        # 7x7 convolution results
        [x > 0 for x in x7R];
        [x > x7_hist_pos_25R for x in x7R];
        [x > x7_hist_pos_34R for x in x7R];
        [x > x7_hist_pos_50R for x in x7R];
        [x > x7_hist_pos_75R for x in x7R];
        [x < x7_hist_neg_25R for x in x7R];
        [x < x7_hist_neg_34R for x in x7R];
        [x < x7_hist_neg_50R for x in x7R];
        [x < x7_hist_neg_75R for x in x7R];

        [x > 0 for x in y7R];
        [x > y7_hist_pos_25R for x in y7R];
        [x > y7_hist_pos_34R for x in y7R];
        [x > y7_hist_pos_50R for x in y7R];
        [x > y7_hist_pos_75R for x in y7R];
        [x < y7_hist_neg_25R for x in y7R];
        [x < y7_hist_neg_34R for x in y7R];
        [x < y7_hist_neg_50R for x in y7R];
        [x < y7_hist_neg_75R for x in y7R];

        # 9x9 convolution results
        [x > 0 for x in x9R];
        [x > x9_hist_pos_25R for x in x9R];
        [x > x9_hist_pos_34R for x in x9R];
        [x > x9_hist_pos_50R for x in x9R];
        [x > x9_hist_pos_75R for x in x9R];
        [x < x9_hist_neg_25R for x in x9R];
        [x < x9_hist_neg_34R for x in x9R];
        [x < x9_hist_neg_50R for x in x9R];
        [x < x9_hist_neg_75R for x in x9R];

        [x > 0 for x in y9R];
        [x > y9_hist_pos_25R for x in y9R];
        [x > y9_hist_pos_34R for x in y9R];
        [x > y9_hist_pos_50R for x in y9R];
        [x > y9_hist_pos_75R for x in y9R];
        [x < y9_hist_neg_25R for x in y9R];
        [x < y9_hist_neg_34R for x in y9R];
        [x < y9_hist_neg_50R for x in y9R];
        [x < y9_hist_neg_75R for x in y9R];


        # Raw pixels
        [x > 0 for x in rawG];
        [x > raw_hist_25G for x in rawG];
        [x > raw_hist_50G for x in rawG];
        [x > raw_hist_75G for x in rawG];

        # 3x3 convolution results
        [x > 0 for x in x3G];
        [x > x3_hist_pos_25G for x in x3G];
        [x > x3_hist_pos_34G for x in x3G];
        [x > x3_hist_pos_50G for x in x3G];
        [x > x3_hist_pos_75G for x in x3G];
        [x < x3_hist_neg_25G for x in x3G];
        [x < x3_hist_neg_34G for x in x3G];
        [x < x3_hist_neg_50G for x in x3G];
        [x < x3_hist_neg_75G for x in x3G];

        [x > 0 for x in y3G];
        [x > y3_hist_pos_25G for x in y3G];
        [x > y3_hist_pos_34G for x in y3G];
        [x > y3_hist_pos_50G for x in y3G];
        [x > y3_hist_pos_75G for x in y3G];
        [x < y3_hist_neg_25G for x in y3G];
        [x < y3_hist_neg_34G for x in y3G];
        [x < y3_hist_neg_50G for x in y3G];
        [x < y3_hist_neg_75G for x in y3G];

        # 5x5 convolution results
        [x > 0 for x in x5G];
        [x > x5_hist_pos_25G for x in x5G];
        [x > x5_hist_pos_34G for x in x5G];
        [x > x5_hist_pos_50G for x in x5G];
        [x > x5_hist_pos_75G for x in x5G];
        [x < x5_hist_neg_25G for x in x5G];
        [x < x5_hist_neg_34G for x in x5G];
        [x < x5_hist_neg_50G for x in x5G];
        [x < x5_hist_neg_75G for x in x5G];

        [x > 0 for x in y5G];
        [x > y5_hist_pos_25G for x in y5G];
        [x > y5_hist_pos_34G for x in y5G];
        [x > y5_hist_pos_50G for x in y5G];
        [x > y5_hist_pos_75G for x in y5G];
        [x < y5_hist_neg_25G for x in y5G];
        [x < y5_hist_neg_34G for x in y5G];
        [x < y5_hist_neg_50G for x in y5G];
        [x < y5_hist_neg_75G for x in y5G];

        # 7x7 convolution results
        [x > 0 for x in x7G];
        [x > x7_hist_pos_25G for x in x7G];
        [x > x7_hist_pos_34G for x in x7G];
        [x > x7_hist_pos_50G for x in x7G];
        [x > x7_hist_pos_75G for x in x7G];
        [x < x7_hist_neg_25G for x in x7G];
        [x < x7_hist_neg_34G for x in x7G];
        [x < x7_hist_neg_50G for x in x7G];
        [x < x7_hist_neg_75G for x in x7G];

        [x > 0 for x in y7G];
        [x > y7_hist_pos_25G for x in y7G];
        [x > y7_hist_pos_34G for x in y7G];
        [x > y7_hist_pos_50G for x in y7G];
        [x > y7_hist_pos_75G for x in y7G];
        [x < y7_hist_neg_25G for x in y7G];
        [x < y7_hist_neg_34G for x in y7G];
        [x < y7_hist_neg_50G for x in y7G];
        [x < y7_hist_neg_75G for x in y7G];

        # 9x9 convolution results
        [x > 0 for x in x9G];
        [x > x9_hist_pos_25G for x in x9G];
        [x > x9_hist_pos_34G for x in x9G];
        [x > x9_hist_pos_50G for x in x9G];
        [x > x9_hist_pos_75G for x in x9G];
        [x < x9_hist_neg_25G for x in x9G];
        [x < x9_hist_neg_34G for x in x9G];
        [x < x9_hist_neg_50G for x in x9G];
        [x < x9_hist_neg_75G for x in x9G];

        [x > 0 for x in y9G];
        [x > y9_hist_pos_25G for x in y9G];
        [x > y9_hist_pos_34G for x in y9G];
        [x > y9_hist_pos_50G for x in y9G];
        [x > y9_hist_pos_75G for x in y9G];
        [x < y9_hist_neg_25G for x in y9G];
        [x < y9_hist_neg_34G for x in y9G];
        [x < y9_hist_neg_50G for x in y9G];
        [x < y9_hist_neg_75G for x in y9G];


        # Raw pixels
        [x > 0 for x in rawB];
        [x > raw_hist_25B for x in rawB];
        [x > raw_hist_50B for x in rawB];
        [x > raw_hist_75B for x in rawB];

        # 3x3 convolution results
        [x > 0 for x in x3B];
        [x > x3_hist_pos_25B for x in x3B];
        [x > x3_hist_pos_34B for x in x3B];
        [x > x3_hist_pos_50B for x in x3B];
        [x > x3_hist_pos_75B for x in x3B];
        [x < x3_hist_neg_25B for x in x3B];
        [x < x3_hist_neg_34B for x in x3B];
        [x < x3_hist_neg_50B for x in x3B];
        [x < x3_hist_neg_75B for x in x3B];

        [x > 0 for x in y3B];
        [x > y3_hist_pos_25B for x in y3B];
        [x > y3_hist_pos_34B for x in y3B];
        [x > y3_hist_pos_50B for x in y3B];
        [x > y3_hist_pos_75B for x in y3B];
        [x < y3_hist_neg_25B for x in y3B];
        [x < y3_hist_neg_34B for x in y3B];
        [x < y3_hist_neg_50B for x in y3B];
        [x < y3_hist_neg_75B for x in y3B];

        # 5x5 convolution results
        [x > 0 for x in x5B];
        [x > x5_hist_pos_25B for x in x5B];
        [x > x5_hist_pos_34B for x in x5B];
        [x > x5_hist_pos_50B for x in x5B];
        [x > x5_hist_pos_75B for x in x5B];
        [x < x5_hist_neg_25B for x in x5B];
        [x < x5_hist_neg_34B for x in x5B];
        [x < x5_hist_neg_50B for x in x5B];
        [x < x5_hist_neg_75B for x in x5B];

        [x > 0 for x in y5B];
        [x > y5_hist_pos_25B for x in y5B];
        [x > y5_hist_pos_34B for x in y5B];
        [x > y5_hist_pos_50B for x in y5B];
        [x > y5_hist_pos_75B for x in y5B];
        [x < y5_hist_neg_25B for x in y5B];
        [x < y5_hist_neg_34B for x in y5B];
        [x < y5_hist_neg_50B for x in y5B];
        [x < y5_hist_neg_75B for x in y5B];

        # 7x7 convolution results
        [x > 0 for x in x7B];
        [x > x7_hist_pos_25B for x in x7B];
        [x > x7_hist_pos_34B for x in x7B];
        [x > x7_hist_pos_50B for x in x7B];
        [x > x7_hist_pos_75B for x in x7B];
        [x < x7_hist_neg_25B for x in x7B];
        [x < x7_hist_neg_34B for x in x7B];
        [x < x7_hist_neg_50B for x in x7B];
        [x < x7_hist_neg_75B for x in x7B];

        [x > 0 for x in y7B];
        [x > y7_hist_pos_25B for x in y7B];
        [x > y7_hist_pos_34B for x in y7B];
        [x > y7_hist_pos_50B for x in y7B];
        [x > y7_hist_pos_75B for x in y7B];
        [x < y7_hist_neg_25B for x in y7B];
        [x < y7_hist_neg_34B for x in y7B];
        [x < y7_hist_neg_50B for x in y7B];
        [x < y7_hist_neg_75B for x in y7B];

        # 9x9 convolution results
        [x > 0 for x in x9B];
        [x > x9_hist_pos_25B for x in x9B];
        [x > x9_hist_pos_34B for x in x9B];
        [x > x9_hist_pos_50B for x in x9B];
        [x > x9_hist_pos_75B for x in x9B];
        [x < x9_hist_neg_25B for x in x9B];
        [x < x9_hist_neg_34B for x in x9B];
        [x < x9_hist_neg_50B for x in x9B];
        [x < x9_hist_neg_75B for x in x9B];

        [x > 0 for x in y9B];
        [x > y9_hist_pos_25B for x in y9B];
        [x > y9_hist_pos_34B for x in y9B];
        [x > y9_hist_pos_50B for x in y9B];
        [x > y9_hist_pos_75B for x in y9B];
        [x < y9_hist_neg_25B for x in y9B];
        [x < y9_hist_neg_34B for x in y9B];
        [x < y9_hist_neg_50B for x in y9B];
        [x < y9_hist_neg_75B for x in y9B];
    ]))
end

const input_size = bools(
    x_trainR[1],
    x_train_conv_orient_x3R[1],
    x_train_conv_orient_y3R[1],
    x_train_conv_orient_x5R[1],
    x_train_conv_orient_y5R[1],
    x_train_conv_orient_x7R[1],
    x_train_conv_orient_y7R[1],
    x_train_conv_orient_x9R[1],
    x_train_conv_orient_y9R[1],
    x_trainG[1],
    x_train_conv_orient_x3G[1],
    x_train_conv_orient_y3G[1],
    x_train_conv_orient_x5G[1],
    x_train_conv_orient_y5G[1],
    x_train_conv_orient_x7G[1],
    x_train_conv_orient_y7G[1],
    x_train_conv_orient_x9G[1],
    x_train_conv_orient_y9G[1],
    x_trainB[1],
    x_train_conv_orient_x3B[1],
    x_train_conv_orient_y3B[1],
    x_train_conv_orient_x5B[1],
    x_train_conv_orient_y5B[1],
    x_train_conv_orient_x7B[1],
    x_train_conv_orient_y7B[1],
    x_train_conv_orient_x9B[1],
    x_train_conv_orient_y9B[1],
).len

X_train = InputBatch(input_size, length(x_train))
@threads for i in eachindex(x_train)
    bv = bools(
        x_trainR[i],
        x_train_conv_orient_x3R[i],
        x_train_conv_orient_y3R[i],
        x_train_conv_orient_x5R[i],
        x_train_conv_orient_y5R[i],
        x_train_conv_orient_x7R[i],
        x_train_conv_orient_y7R[i],
        x_train_conv_orient_x9R[i],
        x_train_conv_orient_y9R[i],
        x_trainG[i],
        x_train_conv_orient_x3G[i],
        x_train_conv_orient_y3G[i],
        x_train_conv_orient_x5G[i],
        x_train_conv_orient_y5G[i],
        x_train_conv_orient_x7G[i],
        x_train_conv_orient_y7G[i],
        x_train_conv_orient_x9G[i],
        x_train_conv_orient_y9G[i],
        x_trainB[i],
        x_train_conv_orient_x3B[i],
        x_train_conv_orient_y3B[i],
        x_train_conv_orient_x5B[i],
        x_train_conv_orient_y5B[i],
        x_train_conv_orient_x7B[i],
        x_train_conv_orient_y7B[i],
        x_train_conv_orient_x9B[i],
        x_train_conv_orient_y9B[i],
    )
    copyto!(view(X_train, :, i), InputVector(bv, copy=false))
end

X_test = InputBatch(input_size, length(x_test))
@threads for i in eachindex(x_test)
    bv = bools(
        x_testR[i],
        x_test_conv_orient_x3R[i],
        x_test_conv_orient_y3R[i],
        x_test_conv_orient_x5R[i],
        x_test_conv_orient_y5R[i],
        x_test_conv_orient_x7R[i],
        x_test_conv_orient_y7R[i],
        x_test_conv_orient_x9R[i],
        x_test_conv_orient_y9R[i],
        x_testG[i],
        x_test_conv_orient_x3G[i],
        x_test_conv_orient_y3G[i],
        x_test_conv_orient_x5G[i],
        x_test_conv_orient_y5G[i],
        x_test_conv_orient_x7G[i],
        x_test_conv_orient_y7G[i],
        x_test_conv_orient_x9G[i],
        x_test_conv_orient_y9G[i],
        x_testB[i],
        x_test_conv_orient_x3B[i],
        x_test_conv_orient_y3B[i],
        x_test_conv_orient_x5B[i],
        x_test_conv_orient_y5B[i],
        x_test_conv_orient_x7B[i],
        x_test_conv_orient_y7B[i],
        x_test_conv_orient_x9B[i],
        x_test_conv_orient_y9B[i],
    )
    copyto!(view(X_test, :, i), InputVector(bv, copy=false))
end

y_train = Int8.(y_train)
y_test = Int8.(y_test)

Serialization.serialize(joinpath(tempdir(), "CIFAR10_train"), (X_train, y_train))
Serialization.serialize(joinpath(tempdir(), "CIFAR10_test"), (X_test, y_test))
Serialization.serialize(joinpath(tempdir(), "CIFAR10_input_size"), input_size)

println("Done.")
