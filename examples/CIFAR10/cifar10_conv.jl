include("../../src/Tsetlin.jl")


using Serialization
using .Tsetlin: TMClassifier, train!, save, load, benchmark, compile


X_train, y_train = Serialization.deserialize("/tmp/CIFAR10_train")
X_test, y_test = Serialization.deserialize("/tmp/CIFAR10_test")


CLAUSES = 20  # (69%+ acc)
T = 1600
S = 1000
L = 4000
LF = 4000

# CLAUSES = 20
# T = 45
# S = 1000
# L = 200
# LF = 200

# CLAUSES = 200
# T = 316
# S = 1000
# L = 1000
# LF = 1000

# CLAUSES = 200
# T = 2500
# S = 1000
# L = 1000
# LF = 1000

# CLAUSES = 2000
# T = 10000  # 2200
# S = 1000   # 1000
# L = 1000   # 200
# LF = 1000  # 200

# CLAUSES = 2000
# T = 2200
# S = 1000
# L = 200
# LF = 200

# CLAUSES = 2000
# T = 4000
# S = 1000
# L = 250
# LF = 250

EPOCHS = 200

# Training the TM model
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=240)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, shuffle=true, verbose=1, index=true)

save(tm, "/tmp/tm.tm")
tm = load("/tmp/tm.tm")
tmc = compile(tm)

benchmark(tmc, X_test, y_test, 10 * 2, warmup=true, index=false)
