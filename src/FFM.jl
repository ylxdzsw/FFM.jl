using OhMyJulia
using Knet
using JLD2

export FFM

struct FFM{T}
    w0::T
    w::Vector{T}
    V::Matrix{Vector{T}}
end

function FFM{T}(n, nf, k=4)
    FFM{T}(0, rand(T, n), [rand(T, k) for i in n, j in nf])
end

function _predict(m, X::Vector{Tuple{Int, Int}})
    w0, w, V = m
    n = length(X)

    s = w0 + mapreduce(x->w[car(x)], +, X)
    for k1 in 1:n, k2 in k1+1:n
        i, fi, xi = X[k1]
        j, fj, xj = X[k2]

        s += V[i, fj]' * V[j, fi]
    end

    s
end

function loss(m, X, y)
    .5(_predict(m, X) - y) ^ 2
end

function train(m::FFM, Xs, ys, o)
    m = m.w0, m.w, m.V
    g = grad(loss)

    o = optimizers(m, o)
    ∇ = mapreduce(x->g(m, car(x), cadr(x)), (a,b)->a.+b, zip(Xs, ys))
    update!(m, ∇, o)


end
