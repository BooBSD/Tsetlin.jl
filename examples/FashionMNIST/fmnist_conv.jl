include("../../src/Tsetlin.jl")
include("../../src/utils/fastconv.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Base.Threads
using Serialization
using Statistics
using MLDatasets: FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, unzip, predict


x_train, y_train = unzip([FashionMNIST(:train)...])
x_test, y_test = unzip([FashionMNIST(:test)...])

print("Preparing input data... ")

# Convolution kernels
Kx3 = [-1 0 1; -2 0 2; -1 0 1] * one(Float32)
#Kx5 = [-2 -1 0 1 2; -3 -2 0 2 3; -4 -3 0 3 4; -3 -2 0 2 3; -2 -1 0 1 2] * one(Float32)
Kx7 = [-3 -2 -1 0 1 2 3; -4 -3 -2 0 2 3 4; -5 -4 -3 0 3 4 5; -6 -5 -4 0 4 5 6; -5 -4 -3 0 3 4 5; -4 -3 -2 0 2 3 4; -3 -2 -1 0 1 2 3] * one(Float32)

#Kx3 = [0 1 2; -1 0 1; -2 -1 0] * one(Float32)
Kx5 = [0 1 2 3 4; -1 0 2 3 3; -2 -2 0 2 2; -3 -3 -2 0 1; -4 -3 -2 -1 0] * one(Float32)
#Kx7 = [0 1 2 3 4 5 6; -1 0 2 3 4 5 5; -2 -2 0 3 4 4 4; -3 -3 -3 0 3 3 3; -4 -4 -4 -3 0 2 2; -5 -5 -4 -3 -2 0 1; -6 -5 -4 -3 -2 -1 0] * one(Float32)

Kx9 = [-1 -1 -1; 2 2 2; -1 -1 -1] * one(Float32)


Ky3 = rotl90(Kx3)
Ky5 = rotl90(Kx5)
Ky7 = rotl90(Kx7)
Ky9 = rotl90(Kx9)

Kp3 = 1  # Padding 1
Kp5 = 2  # Padding 2
Kp7 = 3  # Padding 3
Kp9 = 1  # Padding 1

x_train_conv_orient_x3 = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_train]
x_train_conv_orient_y3 = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_train]

x_test_conv_orient_x3 = [fastconv(x, Kx3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_test]
x_test_conv_orient_y3 = [fastconv(x, Ky3)[1+Kp3:end-Kp3, 1+Kp3:end-Kp3] for x in x_test]

x_train_conv_orient_x5 = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_train]
x_train_conv_orient_y5 = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_train]

x_test_conv_orient_x5 = [fastconv(x, Kx5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_test]
x_test_conv_orient_y5 = [fastconv(x, Ky5)[1+Kp5:end-Kp5, 1+Kp5:end-Kp5] for x in x_test]

x_train_conv_orient_x7 = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_train]
x_train_conv_orient_y7 = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_train]

x_test_conv_orient_x7 = [fastconv(x, Kx7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_test]
x_test_conv_orient_y7 = [fastconv(x, Ky7)[1+Kp7:end-Kp7, 1+Kp7:end-Kp7] for x in x_test]

x_train_conv_orient_x9 = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_train]
x_train_conv_orient_y9 = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_train]

x_test_conv_orient_x9 = [fastconv(x, Kx9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_test]
x_test_conv_orient_y9 = [fastconv(x, Ky9)[1+Kp9:end-Kp9, 1+Kp9:end-Kp9] for x in x_test]

# Booleanization
function bools(raw, x3, y3, x5, y5, x7, y7, x9, y9)
    raw_hist_25::Float64 = quantile([x for x in raw if x > 0], 0.25)
    raw_hist_50::Float64 = quantile([x for x in raw if x > 0], 0.50)
    raw_hist_75::Float64= quantile([x for x in raw if x > 0], 0.75)

    x3_hist_pos_25::Float64 = quantile([x for x in x3 if x > 0], 0.25)
    x3_hist_pos_34::Float64 = quantile([x for x in x3 if x > 0], 0.34)
    x3_hist_pos_50::Float64 = quantile([x for x in x3 if x > 0], 0.50)
    x3_hist_pos_75::Float64 = quantile([x for x in x3 if x > 0], 0.75)
    x3_hist_neg_25::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.25)
    x3_hist_neg_34::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.34)
    x3_hist_neg_50::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.50)
    x3_hist_neg_75::Float64 = quantile([x for x in x3 if x < 0], 1 - 0.75)

    y3_hist_pos_25::Float64 = quantile([x for x in y3 if x > 0], 0.25)
    y3_hist_pos_34::Float64 = quantile([x for x in y3 if x > 0], 0.34)
    y3_hist_pos_50::Float64 = quantile([x for x in y3 if x > 0], 0.50)
    y3_hist_pos_75::Float64 = quantile([x for x in y3 if x > 0], 0.75)
    y3_hist_neg_25::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.25)
    y3_hist_neg_34::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.34)
    y3_hist_neg_50::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.50)
    y3_hist_neg_75::Float64 = quantile([x for x in y3 if x < 0], 1 - 0.75)

    x5_hist_pos_25::Float64 = quantile([x for x in x5 if x > 0], 0.25)
    x5_hist_pos_34::Float64 = quantile([x for x in x5 if x > 0], 0.34)
    x5_hist_pos_50::Float64 = quantile([x for x in x5 if x > 0], 0.50)
    x5_hist_pos_75::Float64 = quantile([x for x in x5 if x > 0], 0.75)
    x5_hist_neg_25::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.25)
    x5_hist_neg_34::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.34)
    x5_hist_neg_50::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.50)
    x5_hist_neg_75::Float64 = quantile([x for x in x5 if x < 0], 1 - 0.75)

    y5_hist_pos_25::Float64 = quantile([x for x in y5 if x > 0], 0.25)
    y5_hist_pos_34::Float64 = quantile([x for x in y5 if x > 0], 0.34)
    y5_hist_pos_50::Float64 = quantile([x for x in y5 if x > 0], 0.50)
    y5_hist_pos_75::Float64 = quantile([x for x in y5 if x > 0], 0.75)
    y5_hist_neg_25::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.25)
    y5_hist_neg_34::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.34)
    y5_hist_neg_50::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.50)
    y5_hist_neg_75::Float64 = quantile([x for x in y5 if x < 0], 1 - 0.75)

    x7_hist_pos_25::Float64 = quantile([x for x in x7 if x > 0], 0.25)
    x7_hist_pos_34::Float64 = quantile([x for x in x7 if x > 0], 0.34)
    x7_hist_pos_50::Float64 = quantile([x for x in x7 if x > 0], 0.50)
    x7_hist_pos_75::Float64 = quantile([x for x in x7 if x > 0], 0.75)
    x7_hist_neg_25::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.25)
    x7_hist_neg_34::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.34)
    x7_hist_neg_50::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.50)
    x7_hist_neg_75::Float64 = quantile([x for x in x7 if x < 0], 1 - 0.75)

    y7_hist_pos_25::Float64 = quantile([x for x in y7 if x > 0], 0.25)
    y7_hist_pos_34::Float64 = quantile([x for x in y7 if x > 0], 0.34)
    y7_hist_pos_50::Float64 = quantile([x for x in y7 if x > 0], 0.50)
    y7_hist_pos_75::Float64 = quantile([x for x in y7 if x > 0], 0.75)
    y7_hist_neg_25::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.25)
    y7_hist_neg_34::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.34)
    y7_hist_neg_50::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.50)
    y7_hist_neg_75::Float64 = quantile([x for x in y7 if x < 0], 1 - 0.75)

    x9_hist_pos_25::Float64 = quantile([x for x in x9 if x > 0], 0.25)
    x9_hist_pos_34::Float64 = quantile([x for x in x9 if x > 0], 0.34)
    x9_hist_pos_50::Float64 = quantile([x for x in x9 if x > 0], 0.50)
    x9_hist_pos_75::Float64 = quantile([x for x in x9 if x > 0], 0.75)
    x9_hist_neg_25::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.25)
    x9_hist_neg_34::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.34)
    x9_hist_neg_50::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.50)
    x9_hist_neg_75::Float64 = quantile([x for x in x9 if x < 0], 1 - 0.75)

    y9_hist_pos_25::Float64 = quantile([x for x in y9 if x > 0], 0.25)
    y9_hist_pos_34::Float64 = quantile([x for x in y9 if x > 0], 0.34)
    y9_hist_pos_50::Float64 = quantile([x for x in y9 if x > 0], 0.50)
    y9_hist_pos_75::Float64 = quantile([x for x in y9 if x > 0], 0.75)
    y9_hist_neg_25::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.25)
    y9_hist_neg_34::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.34)
    y9_hist_neg_50::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.50)
    y9_hist_neg_75::Float64 = quantile([x for x in y9 if x < 0], 1 - 0.75)

    return TMInput([
        # Raw pixels
        [x > 0 for x in raw];
        [x > raw_hist_25 for x in raw];
        [x > raw_hist_50 for x in raw];
        [x > raw_hist_75 for x in raw];

        # 3x3 convolution results
        [x > 0 for x in x3];
        [x > x3_hist_pos_25 for x in x3];
        [x > x3_hist_pos_34 for x in x3];
        [x > x3_hist_pos_50 for x in x3];
        [x > x3_hist_pos_75 for x in x3];
        [x < x3_hist_neg_25 for x in x3];
        [x < x3_hist_neg_34 for x in x3];
        [x < x3_hist_neg_50 for x in x3];
        [x < x3_hist_neg_75 for x in x3];

        [x > 0 for x in y3];
        [x > y3_hist_pos_25 for x in y3];
        [x > y3_hist_pos_34 for x in y3];
        [x > y3_hist_pos_50 for x in y3];
        [x > y3_hist_pos_75 for x in y3];
        [x < y3_hist_neg_25 for x in y3];
        [x < y3_hist_neg_34 for x in y3];
        [x < y3_hist_neg_50 for x in y3];
        [x < y3_hist_neg_75 for x in y3];

        # 5x5 convolution results
        [x > 0 for x in x5];
        [x > x5_hist_pos_25 for x in x5];
        [x > x5_hist_pos_34 for x in x5];
        [x > x5_hist_pos_50 for x in x5];
        [x > x5_hist_pos_75 for x in x5];
        [x < x5_hist_neg_25 for x in x5];
        [x < x5_hist_neg_34 for x in x5];
        [x < x5_hist_neg_50 for x in x5];
        [x < x5_hist_neg_75 for x in x5];

        [x > 0 for x in y5];
        [x > y5_hist_pos_25 for x in y5];
        [x > y5_hist_pos_34 for x in y5];
        [x > y5_hist_pos_50 for x in y5];
        [x > y5_hist_pos_75 for x in y5];
        [x < y5_hist_neg_25 for x in y5];
        [x < y5_hist_neg_34 for x in y5];
        [x < y5_hist_neg_50 for x in y5];
        [x < y5_hist_neg_75 for x in y5];

        # 7x7 convolution results
        [x > 0 for x in x7];
        [x > x7_hist_pos_25 for x in x7];
        [x > x7_hist_pos_34 for x in x7];
        [x > x7_hist_pos_50 for x in x7];
        [x > x7_hist_pos_75 for x in x7];
        [x < x7_hist_neg_25 for x in x7];
        [x < x7_hist_neg_34 for x in x7];
        [x < x7_hist_neg_50 for x in x7];
        [x < x7_hist_neg_75 for x in x7];

        [x > 0 for x in y7];
        [x > y7_hist_pos_25 for x in y7];
        [x > y7_hist_pos_34 for x in y7];
        [x > y7_hist_pos_50 for x in y7];
        [x > y7_hist_pos_75 for x in y7];
        [x < y7_hist_neg_25 for x in y7];
        [x < y7_hist_neg_34 for x in y7];
        [x < y7_hist_neg_50 for x in y7];
        [x < y7_hist_neg_75 for x in y7];

        # 9x9 convolution results
        [x > 0 for x in x9];
        [x > x9_hist_pos_25 for x in x9];
        [x > x9_hist_pos_34 for x in x9];
        [x > x9_hist_pos_50 for x in x9];
        [x > x9_hist_pos_75 for x in x9];
        [x < x9_hist_neg_25 for x in x9];
        [x < x9_hist_neg_34 for x in x9];
        [x < x9_hist_neg_50 for x in x9];
        [x < x9_hist_neg_75 for x in x9];

        [x > 0 for x in y9];
        [x > y9_hist_pos_25 for x in y9];
        [x > y9_hist_pos_34 for x in y9];
        [x > y9_hist_pos_50 for x in y9];
        [x > y9_hist_pos_75 for x in y9];
        [x < y9_hist_neg_25 for x in y9];
        [x < y9_hist_neg_34 for x in y9];
        [x < y9_hist_neg_50 for x in y9];
        [x < y9_hist_neg_75 for x in y9];
    ])
end


X_train::Vector{TMInput} = Vector{TMInput}(undef, length(x_train))
@threads for i in eachindex(x_train)
    X_train[i] = bools(x_train[i], x_train_conv_orient_x3[i], x_train_conv_orient_y3[i], x_train_conv_orient_x5[i], x_train_conv_orient_y5[i], x_train_conv_orient_x7[i], x_train_conv_orient_y7[i], x_train_conv_orient_x9[i], x_train_conv_orient_y9[i])
end

X_test::Vector{TMInput} = Vector{TMInput}(undef, length(x_test))
@threads for i in eachindex(x_test)
    X_test[i] = bools(x_test[i], x_test_conv_orient_x3[i], x_test_conv_orient_y3[i], x_test_conv_orient_x5[i], x_test_conv_orient_y5[i], x_test_conv_orient_x7[i], x_test_conv_orient_y7[i], x_test_conv_orient_x9[i], x_test_conv_orient_y9[i])
end

y_train = Int8.(y_train)
y_test = Int8.(y_test)

println("Done.")

# CLAUSES = 2  # Best accuracy: 92.53% after 713 epochs
# T = 80
# S = 1000
# L = 1200
# LF = 1200

CLAUSES = 20  # Best accuracy: 93.59% after 857 epochs
T = 100
S = 700
L = 200
LF = 200

EPOCHS = 1000

# Training the TM model
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=230)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, shuffle=true, verbose=1, index=false)
