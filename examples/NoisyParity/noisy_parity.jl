include("../../src/Tsetlin.jl")


using Base.Threads
using Serialization
using .Tsetlin: TMInput, TMClassifier, train!, save


train = readlines("/tmp/NoisyParityTrainingData.txt")
test = readlines("/tmp/NoisyParityTestingData.txt")

X_train::Vector = Vector{TMInput}(undef, length(train))
y_train::Vector = Vector{Int8}(undef, length(train))
@threads for i in eachindex(train)
    data = [parse(Bool, x) for x in split(train[i], " ")]
    X_train[i] = TMInput(@views(data[1:end-1]))
    y_train[i] = Int8(last(data))
end

X_test::Vector = Vector{TMInput}(undef, length(test))
y_test::Vector = Vector{Int8}(undef, length(test))
@threads for i in eachindex(test)
    data = [parse(Bool, x) for x in split(test[i], " ")]
    X_test[i] = TMInput(@views(data[1:end-1]))
    y_test[i] = Int8(last(data))
end


# CLAUSES = 16
# T = 7
# S = 12
# L = 8
# LF = 4

CLAUSES = 200
T = 20
S = 12
L = 8
LF = 4

EPOCHS = 1000

# Training the TM model
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=220)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, verbose=1)
