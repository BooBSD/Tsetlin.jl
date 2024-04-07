include("../src/Tsetlin.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, combine, benchmark, save, load, optimize!, unzip


x_train, y_train = unzip([m for m in MNIST(split=:train)])
x_test, y_test = unzip([m for m in MNIST(split=:test)])
# x_train, y_train = unzip([m for m in FashionMNIST(split=:train)])
# x_test, y_test = unzip([m for m in FashionMNIST(split=:test)])

# Booleanization
x_train = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.33 true else false end for x in i];
    [if x > 0.66 true else false end for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.33 true else false end for x in i];
    [if x > 0.66 true else false end for x in i];
])) for i in x_test]

const EPOCHS = 500
const CLAUSES = 2048
const T = 32
const R = 0.94
const L = 12
const best_tms_size = 500

# Training the TM model
tm = TMClassifier(CLAUSES, T, R, L=L, states_num=256, include_limit=128)
_, tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, best_tms_compile=true, shuffle=true, batch=true)

save(tms, "/tmp/tms.tm")
tms = load("/tmp/tms.tm")

# Binomial combining of trained TM models
_, tm = combine(tms[1:500], 2, x_test, y_test, batch=true)
save(tm, "/tmp/tm2.tm")
tm = load("/tmp/tm2.tm")

# Optimizing the TM model
optimize!(tm, x_train)
save(tm, "/tmp/tm_optimized.tm")
tm_opt = load("/tmp/tm_optimized.tm")

benchmark(tm_opt, x_test, y_test, 6400, batch=true, warmup=true, deep_copy=true)
