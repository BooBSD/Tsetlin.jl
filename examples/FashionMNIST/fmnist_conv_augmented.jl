include("../../src/Tsetlin.jl")

using Serialization
using .Tsetlin: TMClassifier, train!


X_train, y_train = Serialization.deserialize("/tmp/FMNIST_train")
X_test, y_test = Serialization.deserialize("/tmp/FMNIST_test")

# CLAUSES = 20
# T = 200
# S = 500
# L = 1000
# LF = 1000

CLAUSES = 200  # acc: 94.49% after 40 epochs
T = 282 * 4
S = 1000
L = 1000
LF = 800

# CLAUSES = 8000  # Best accuracy: 94.74% after 11 epochs, Normal 94.68% test acc after 50 epochs.
# T = 700
# S = 700
# L = 30
# LF = 30

EPOCHS = 200

# Training the TM model
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=240)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, shuffle=true, verbose=1, index=false)
