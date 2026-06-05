include("../../src/Tsetlin.jl")
include("../../src/utils/explain.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["CairoMakie"]]

using MLDatasets: MNIST, FashionMNIST
using CairoMakie
using .Tsetlin: TMInput, TMClassifier, train!, save, load, unzip, booleanize, compile, predict, accuracy


MODEL_PATH = joinpath(tempdir(), "tm.tm")
IMAGE_PATH = joinpath(tempdir(), "FPTM_MNIST_heatmap.png")

x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# 1-bit booleanization
x_train = [booleanize(x, 0.2) for x in x_train]
x_test = [booleanize(x, 0.2) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)


CLAUSES = 20
T = 20
S = 400
L = 150
LF = 75

# CLAUSES = 512
# T = 45
# S = 200
# L = 16
# LF = 8

EPOCHS = 1000

# Training the TM model
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L, LF, states_num=256, include_limit=128)
train!(tm, x_train, y_train, x_test, y_test, EPOCHS, index=false)

# Compiling model
tmc = compile(tm)

save(tmc, MODEL_PATH)
tmc = load(MODEL_PATH)


function draw_heatmap!(ax, literals::Vector{Int16}; colormap = :hot)
    data = reshape(literals, 28, 28)
    data = reverse(data, dims=2)
    heatmap!(ax, data, colormap=colormap)
    hidedecorations!(ax)
    hidespines!(ax)
end


function draw_block!(fig, start_row, vectors, title; colormap = :hot)
    for i in eachindex(vectors)
        row = start_row + 1 + (i - 1) ÷ 5
        col = (i - 1) % 5 + 1
        ax = Axis(fig[row, col], aspect = DataAspect())
        draw_heatmap!(ax, vectors[i])
    end
    Label(fig[start_row, :], title, fontsize=24, halign = :left)
end


function plot_heatmaps(explained_model::Dict; colormap = :hot)
    set_theme!(theme_dark())
    fig = Figure(size = (1000, 460 * 4))
    positive_included_literals = [explained_model[Int8(digit)].positive_included_literals for digit in 0:9]
    positive_included_literals_inverted = [explained_model[Int8(digit)].positive_included_literals_inverted for digit in 0:9]
    negative_included_literals = [explained_model[Int8(digit)].negative_included_literals for digit in 0:9]
    negative_included_literals_inverted = [explained_model[Int8(digit)].negative_included_literals_inverted for digit in 0:9]
    draw_block!(fig, 1,  positive_included_literals, "Positive included literals")
    draw_block!(fig, 4,  positive_included_literals_inverted, "Positive included literals inverted")
    draw_block!(fig, 7,  negative_included_literals, "Negative included literals")
    draw_block!(fig, 10, negative_included_literals_inverted, "Negative included literals inverted")
    Label(fig[0, :], "FPTM MNIST included literal distributed representations", fontsize=32, halign=:left)
    return fig
end


accuracy(predict(tmc, x_test), y_test) |> println

print("Explain model...")
explained_model = explain(tmc)
println("\t\tdone.")
print("Drawing literals heatmap...")
fig = plot_heatmaps(explained_model)
println("\tdone.")
print("Saving image to $(IMAGE_PATH)...")
CairoMakie.save(IMAGE_PATH, fig)
println(" done.")
