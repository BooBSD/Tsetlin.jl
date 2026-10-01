include("../../src/Tsetlin.jl")


using Base.Threads
using .Tsetlin: TMClassifier, InputVector, InputBatch, train!


const STATES_NUM = 256
const INCLUDE_LIMIT = 220
const EPOCHS = 1_000_000

# 8-bit XOR

# const CLAUSES = 32
# const T = 8
# const S = 12
# const L = 8
# const LF = 4

# const CLAUSES = 64
# const T = 12
# const S = 12
# const L = 8
# const LF = 4

const CLAUSES = 128
const T = 16
const S = 12
const L = 8
const LF = 4

# 16-bit XOR

# const CLAUSES = 256
# const T = 32 * 2
# const S = 24
# const L = 16
# const LF = 8

train = readlines(joinpath(tempdir(), "NoisyParityTrainingData.txt"))
test = readlines(joinpath(tempdir(), "NoisyParityTestingData.txt"))

get_input_size(line::String)::Int = length(split(line, " ")) - 1

function prepare_data(lines::Vector{String})::Tuple{InputBatch, Vector{Int8}}
    X = InputBatch(get_input_size(lines[1]), length(lines))
    Y = Vector{Int8}(undef, length(lines))
    @threads for i in eachindex(lines)
        data = [parse(Bool, x) for x in split(lines[i], " ")]
        copyto!(view(X, :, i), InputVector(BitVector(@views(data[1:end-1])), copy=false))
        Y[i] = Int8(last(data))
    end
    return X, Y
end

X_train, y_train = prepare_data(train)
X_test, y_test = prepare_data(test)

# Training the TM model
tm = TMClassifier(get_input_size(train[1]), y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS)
