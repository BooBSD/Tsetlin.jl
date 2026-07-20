include("../../src/Tsetlin.jl")

using Base.Threads
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, benchmark, load, save


# Loading datasets
TM_PATH = joinpath(tempdir(), "tm.tm")
train = readlines(joinpath(tempdir(), "IMDBTrainingData.txt"))
test = readlines(joinpath(tempdir(), "IMDBTestData.txt"))

# Preparing datasets
x_train = Vector{TMInput}(undef, length(train))
y_train = Vector{Bool}(undef, length(train))
@threads for i in eachindex(train)
    xy = [parse(Bool, x) for x in split(train[i], " ")]
    x_train[i] = TMInput(xy[1:length(xy) - 1])
    y_train[i] = xy[length(xy)]
end
x_test = Vector{TMInput}(undef, length(test))
y_test = Vector{Bool}(undef, length(test))
@threads for i in eachindex(test)
    xy = [parse(Bool, x) for x in split(test[i], " ")]
    x_test[i] = TMInput(xy[1:length(xy) - 1])
    y_test[i] = xy[length(xy)]
end

CLAUSES = 1
T = 18
S = 1000
L = 64
LF = 64

EPOCHS = 1000

# Training the TM model
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L, LF, states_num=256, include_limit=220)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=1, index=false)

# Saving model
save(tms[1][1], TM_PATH)
# Loading model
tm = load(TM_PATH)
# Benchmark model
# 135 corresponds to a 5GB input dataset. Feel free to adjust this number if you like.
benchmark(tm, x_test, y_test, 135, index=false)
