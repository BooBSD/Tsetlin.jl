include("../../src/Tsetlin.jl")

using .Tsetlin: TMClassifier, InputVector, InputBatch, train!


const STATES_NUM = 256
const INCLUDE_LIMIT = 240
const EPOCHS = 1000

const CLAUSES = 2000
const T = 100
const S = 1000
const L = 50
const LF = 10

get_input_size(line::String)::Int = length(split(line, " ")) - 1

x_train = readlines(joinpath(tempdir(), "Amazon_X_train.txt"))
y_train = readlines(joinpath(tempdir(), "Amazon_Y_train.txt"))
x_test = readlines(joinpath(tempdir(), "Amazon_X_test.txt"))
y_test = readlines(joinpath(tempdir(), "Amazon_Y_test.txt"))

X_train = InputBatch([InputVector(BitVector(parse(Bool, x) for x in split(X, " ")), copy=true) for X in x_train])
X_test = InputBatch([InputVector(BitVector(parse(Bool, x) for x in split(X, " ")), copy=true) for X in x_test])
y_train = [parse(Int8, Y) for Y in y_train]
y_test = [parse(Int8, Y) for Y in y_test]

# Training the TM model
tm = TMClassifier(get_input_size(x_train[1]), y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS)
