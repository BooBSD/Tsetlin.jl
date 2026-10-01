include("../../src/Tsetlin.jl")

using Base.Threads
using .Tsetlin: TMClassifier, InputVector, InputBatch, train!


const STATES_NUM = 256
const INCLUDE_LIMIT = 240
const EPOCHS = 20

# Optimal hyperparameters:
# const CLAUSES = 200
# const T = 32
# const S = 2000
# const L = 100
# const LF = 10

# Maximum accuracy after 15-20 epochs:
const CLAUSES = 200
const T = 250
const S = 2000
const L = 100
const LF = 10

# Loading datasets
train = readlines(joinpath(tempdir(), "IMDBTrainingData.txt"))
test = readlines(joinpath(tempdir(), "IMDBTestData.txt"))

get_input_size(line::String)::Int = length(split(line, " ")) - 1

function prepare_data(lines::Vector{String})::Tuple{InputBatch, Vector{Int8}}
    X = InputBatch(get_input_size(lines[1]), length(lines))
    Y = Vector{Int8}(undef, length(lines))
    @threads for i in eachindex(lines)
        data = [parse(Bool, x) for x in split(lines[i], " ")]
        copyto!(view(X, :, i), InputVector(BitVector(@views(data[1:end-1])), copy=false))
        Y[i] = last(data)
    end
    return X, Y
end

X_train, y_train = prepare_data(train)
X_test, y_test = prepare_data(test)

# Training the TM model
tm = TMClassifier(get_input_size(train[1]), y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS)
