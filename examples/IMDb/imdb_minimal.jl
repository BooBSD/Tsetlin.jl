include("../../src/Tsetlin.jl")

using Base.Threads
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, benchmark, load, save


# Loading datasets
train = readlines("/tmp/IMDBTrainingData.txt")
test = readlines("/tmp/IMDBTestData.txt")

# Preparing datasets
x_train::Vector{TMInput} = Vector{TMInput}(undef, length(train))
y_train::Vector{Bool} = Vector{Bool}(undef, length(train))
@threads for i in eachindex(train)
    xy = [parse(Bool, x) for x in split(train[i], " ")]
    x_train[i] = TMInput(xy[1:length(xy) - 1])
    y_train[i] = xy[length(xy)]
end
x_test::Vector{TMInput} = Vector{TMInput}(undef, length(test))
y_test::Vector{Bool} = Vector{Bool}(undef, length(test))
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
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=220)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=1, shuffle=true, verbose=1, index=false)

# Saving model
save(tms[1][1], "/tmp/tm.tm")
# Loading model
tm = load("/tmp/tm.tm")
# Benchmark model
# 135 corresponds to a 5GB input dataset. Feel free to adjust this number if you like.
benchmark(tm, x_test, y_test, 135)
