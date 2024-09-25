include("../src/Tsetlin.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, combine, benchmark, save, load, optimize!, unzip


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# Booleanization
x_train = [TMInput([
    x .> 0;
    x .> 0.33;
    x .> 0.66;
]) for x in x_train]
x_test = [TMInput([
    x .> 0;
    x .> 0.33;
    x .> 0.66;
]) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)

# const CLAUSES = 2048
# const T = 32
# const R = 0.94
# const L = 12

const CLAUSES = 128
const T = 8
const R = 0.89
const L = 16

# const CLAUSES = 84
# const T = 5
# const R = 0.85
# const L = 11

const EPOCHS = 2000
const best_tms_size = 512

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, R, L=L, states_num=256, include_limit=128)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=1)

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

benchmark(tm_opt, x_test, y_test, 2000, batch=true, warmup=true, deep_copy=true)
