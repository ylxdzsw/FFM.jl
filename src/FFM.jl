using OhMyJulia
using Knet
using JLD2

struct FFM{T}
    w::Vector{T}
    V::Matrix{Vector{T}}
end

function FFM{T}(n::Integer, nf::Integer, k::Integer=4) where T
    FFM{T}(0 ++ rand(T, n), [rand(T, k) for i in 1:n, j in 1:nf])
end

function predict(m, X::Vector{Tuple{Int, Int}})
    w, V = m
    n = length(X)

    s = car(w) + mapreduce(x->w[car(x) + 1], +, X)
    for k1 in 1:n, k2 in k1+1:n
        i, fi = X[k1]
        j, fj = X[k2]

        # due to a bug of AutoGrad #73 here we need to indexing mannually. and remenber that julia is column-major
        offset = nrow(V)
        s += V[i + offset * (fj - 1)]' * V[j + offset * (fi - 1)]
    end

    s
end
predict(m::FFM, X) = predict((m.w, m.V), X)

function loss(m, X, y)
    .5(predict(m, X) - y) ^ 2
end

# dirty hack
import Base.+
(+)(::Void, ::Void) = nothing
(+)(x, ::Void) = x
(+)(::Void, x) = x

function step!(m::FFM, Xs, ys, o)
    m = m.w, m.V
    g = grad(loss)
    ∇ = mapreduce(x->g(m, car(x), cadr(x)), (a,b)->a.+b, zip(Xs, ys))
    update!(m, ∇, o)
end

function train(train_file, test_file, o=Adam; kwargs...)
    trainX, trainy, n1, nf1 = read_libsvm(train_file)
    testX , testy, n2, nf2  = read_libsvm(test_file)

    n, nf = max(n1, n2), max(nf1, nf2)
    m = FFM{f64}(n, nf)
    o = optimizers((m.w, m.V), o, kwargs...)

    L = mapreduce(x->loss(m, car(x), cadr(x)), +, zip(testX, testy))
    println("epoch: 0, loss: $L")

    for epoch = 1:5
        last = 1
        for i in 10:10:length(trainy)
            step!(m, trainX[last:i], trainy[last:i], o)
            last = i+1
        end

        L = mapreduce(x->loss(m, car(x), cadr(x)), +, zip(testX, testy))
        println("epoch: $epoch, loss: $L")
    end
end

function read_libsvm(file)
    Xs, ys = [], []
    n, nf = Ref(0), Ref(0)
    for line in eachline(split, file)
        push!(ys, parse(Int, car(line)))
        push!(Xs, map(cdr(line)) do x
            field, feature = split(x, ':') # file encoding starts with 0
            field, feature = parse.(Int, (field, feature)) .+ 1
            n[], nf[] = max(n[], feature), max(nf[], field)
            feature, field
        end)
    end
    Xs, ys, n[], nf[]
end
