include("../../src/Tsetlin.jl")

using Serialization
using .Tsetlin: TMClassifier, train!


X_train, y_train = Serialization.deserialize(joinpath(tempdir(), "FMNIST_train"))
X_test, y_test = Serialization.deserialize(joinpath(tempdir(), "FMNIST_test"))
input_size = Serialization.deserialize(joinpath(tempdir(), "FMNIST_input_size"))

# const CLAUSES = 20
# const T = 200
# const S = 500
# const L = 1000
# const LF = 1000

const CLAUSES = 200  # acc: 94.49% after 40 epochs
const T = 282 * 4
const S = 1000
const L = 1000
const LF = 800

# const CLAUSES = 8000  # Best accuracy: 94.74% after 11 epochs, Normal 94.68% test acc after 50 epochs.
# const T = 700
# const S = 700
# const L = 30
# const LF = 30

const STATES_NUM = 256
const INCLUDE_LIMIT = 240
const EPOCHS = 200

# Training the TM model
tm = TMClassifier(input_size, y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS)
