include("../../src/Tsetlin.jl")


using Serialization
using .Tsetlin: TMClassifier, train!


X_train, y_train = Serialization.deserialize(joinpath(tempdir(), "CIFAR10_train"))
X_test, y_test = Serialization.deserialize(joinpath(tempdir(), "CIFAR10_test"))
input_size = Serialization.deserialize(joinpath(tempdir(), "CIFAR10_input_size"))

const CLAUSES = 20  # (69%+ acc)
const T = 1600
const S = 1000
const L = 4000
const LF = 4000

# const CLAUSES = 20
# const T = 45
# const S = 1000
# const L = 200
# const LF = 200

# const CLAUSES = 200
# const T = 316
# const S = 1000
# const L = 1000
# const LF = 1000

# const CLAUSES = 200
# const T = 2500
# const S = 1000
# const L = 1000
# const LF = 1000

# const CLAUSES = 2000
# const T = 10000  # 2200
# const S = 1000   # 1000
# const L = 1000   # 200
# const LF = 1000  # 200

# const CLAUSES = 2000
# const T = 2200
# const S = 1000
# const L = 200
# const LF = 200

# const CLAUSES = 2000
# const T = 4000
# const S = 1000
# const L = 250
# const LF = 250

const STATES_NUM = 256
const INCLUDE_LIMIT = 240
const INDEX = true
const EPOCHS = 1000

# Training the TM model
tm = TMClassifier(input_size, y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS, index=INDEX)
