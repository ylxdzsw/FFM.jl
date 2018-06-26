using OhMyJulia
using JLD2
using FileIO

const ϵ = .1

struct FFM{T<:Real}
    w::Vector{T}
    V::Matrix{Vector{T}}
end

function FFM{T}(n::Int, nf::Int, k::Int=4) where T<:Real
    FFM{T}(0 ++ rand(T, n), [rand(T, k) - .5 for i in 1:n, j in 1:nf])
end

struct Trainer{T<:Real} # RMSProp
    gw::Vector{T}
    gV::Matrix{Vector{T}}
    vw::Vector{T}
    vV::Matrix{Vector{T}}
end

function Trainer(m::FFM{T}) where T
    t = Trainer{T}(deepcopy(m.w), deepcopy(m.V), deepcopy(m.w), deepcopy(m.V))
    reset!(t, true)
    t
end

function reset!(t::Trainer, acc=false)
    t.gw .= 0
    for v in t.gV
        v .= 0
    end

    acc || return

    t.vw .= 0
    for v in t.vV
        v .= 0
    end
end

function predict(m, X)
    n = length(X)

    s = car(m.w) + mapreduce(x->m.w[car(x) + 1], +, X)
    for k1 in 1:n, k2 in k1+1:n
        i, fi = X[k1]
        j, fj = X[k2]

        s += m.V[i, fj]' * m.V[j, fi]
    end

    s
end

function learn!(m, t, X, y)
    Δ = predict(m, X) - y

    t.gw[1] += Δ
    for (i, fi) in X
        t.gw[i+1] += Δ
    end

    n = length(X)
    for k1 in 1:n, k2 in k1+1:n
        i, fi = X[k1]
        j, fj = X[k2]

        t.gV[i, fj] .+= Δ .* m.V[j, fi]
        t.gV[j, fi] .+= Δ .* m.V[i, fj]
    end
end

function step!(m, t, lr=.1, L2=.001, γ = .9)
    @. t.vw = γ * t.vw + (1 - γ) * t.gw ^ 2
    for (vv, gv) in zip(t.vV, t.gV)
        @. vv += γ * vv + (1 - γ) * gv ^ 2
    end

    m.w .*= 1 - L2
    @. m.w -= lr * t.gw / sqrt(t.vw + ϵ)
    for (v, vv, gv) in zip(m.V, t.vV, t.gV)
        v .*= 1 - L2
        @. v -= lr * gv / sqrt(vv + ϵ)
    end

    reset!(t)
end

function loss(m, X, y)
    .5(predict(m, X) - y) ^ 2
end

function train(train_file, test_file)
    trainX, trainy, n1, nf1 = read_libsvm(train_file)
    testX,  testy,  n2, nf2 = read_libsvm(test_file)

    n, nf = max(n1, n2), max(nf1, nf2)
    m = FFM{f64}(n, nf)
    t = Trainer(m)

    L = mapreduce(x->loss(m, car(x), cadr(x)), +, zip(testX, testy))
    println("epoch: 0, loss: $L")

    best = (nothing, Inf)
    for epoch = 1:50
        for i in 1:length(trainy)
            learn!(m, t, trainX[i], trainy[i])
            i % 10 == 0 && step!(m, t)
        end

        L = mapreduce(x->loss(m, car(x), cadr(x)), +, zip(testX, testy))
        best = L < cadr(best) ? (deepcopy(m), L) : best
        println("epoch: $epoch, loss: $L")
    end

    save("models.jld2", Dict("last" => m, "best" => car(best)))
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
