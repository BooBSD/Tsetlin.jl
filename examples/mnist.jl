include("../src/Tsetlin.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, combine, benchmark, save, load, optimize!, unzip, booleanize


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# Booleanizing input data (3 bits per pixel):
x_train = [booleanize(x, 0, 0.33, 0.66) for x in x_train]
x_test = [booleanize(x, 0, 0.33, 0.66) for x in x_test]

# Booleanizing input data (1 bits per pixel) for small model (CLAUSES = 80):
# x_train = [booleanize(x, 0.25) for x in x_train]
# x_test = [booleanize(x, 0.25) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)

CLAUSES = 128
T = 4
S = 24
L = 12

# CLAUSES = 512
# T = 16
# S = 30
# L = 12

# CLAUSES = 80
# T = 3
# S = 15
# L = 6

const EPOCHS = 2000
const best_tms_size = 500

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, states_num=256, include_limit=220)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=2)

save(tms, "/tmp/tms.tm")
tms = load("/tmp/tms.tm")

# Binomial combinatorial merge of trained TM models
tm, _ = combine(tms, 2, x_test, y_test, batch=true)
save(tm, "/tmp/tm2.tm")
tm = load("/tmp/tm2.tm")

# Optimizing the TM model
optimize!(tm, x_train)
save(tm, "/tmp/tm_optimized.tm")
tm_opt = load("/tmp/tm_optimized.tm")

benchmark(tm_opt, x_test, y_test, 2000, batch=true, warmup=true)
