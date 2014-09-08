using NLopt
using ROOT, ROOTHistograms, Histograms;

include("../analysis/hplot.jl")
fpoly(x, vs...) = sum([vs[i] * x.^(i-1) for i=1:length(vs)])

fit_counter = 0
function linear_fit(x::Vector{Float64}, yv::Vector{Float64}, covm::Matrix{Float64})

    xv = x

    function chi2(coefs::Vector{Float64}, grad::Vector{Float64})
        global fit_counter
        z = first(transpose(fpoly(xv, -0.5, coefs[1]) - yv) * inv(covm) * (fpoly(xv, -0.5, coefs[1]) - yv))
        fit_counter += 1
        fit_counter < 10 && println("$fit_counter $z $coefs")
        return z
    end

    priors = [0.0];

    opt = Opt(:LN_COBYLA, length(priors))
    #lower_bounds!(opt, [0, 0])
    #upper_bounds!(opt, [10.0, 10.0])
    xtol_rel!(opt,1e-8)

    min_objective!(opt, chi2)

    (minf,minx,ret) = optimize(opt, priors)
    println("got $minf at $minx after $fit_counter iterations (returned $ret)")
    return minx
end

tf = TFile("histos/mu__nominal.root")
unf = load_with_errors(tf, "unfolded";error_type = :errors)
errs = root_cast(TH2D, Get(tf, "error"))|>from_root

x = edges(unf)[2:end-2]
yv = contents(unf)[2:end-2]
covm = contents(errs)[2:end-2, 2:end-2]

N = sum(yv)
yv = deepcopy(yv/N)
covm = deepcopy(covm/N)

Close(tf)

println(x)
println(yv)
println(covm)

lf = linear_fit(x, yv, covm)

println("fitted $lf")
ax = axes()
#barplot(unf, "black", marker="o")
ax[:errorbar](x, yv, errors(unf)[2:end-2]/N, label="data", marker="o")
ax[:plot](x, fpoly(x, 0.5, lf...), label="fit")
ax[:legend](loc="best")
savefig("test.png")
