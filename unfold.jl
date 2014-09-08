using NLopt
include("../analysis/base.jl");
using SingleTopBase;
using ROOT, ROOTHistograms, Histograms;
using PyCall, PyPlot;

include("$BASE/src/analysis/hplot.jl");
include("$BASE/src/analysis/python_plots.jl");

const hist_path = "../../results/hists/$(ARGS[3])/merged"

titles = {
    :mu=>"\$\\mu^\\pm\$, 2J1T, BDT>0.6",
    :ele=>"\$e^\\pm\$, 2J1T, BDT>0.6",
    :combined=>"\$e^\\pm,\\ \\mu^\\pm\$, 2J1T, BDT>0.6"
};


const toload = ["unfolded", "gen", "correlation", "genrec",  "error"];
const lepton = symbol(ARGS[1])
const unfolded_histos = convert(ASCIIString, ARGS[2])
const FITRESULTS = {
    :mu=>FitResult("$BASE/results/fits/$(ARGS[4])/nominal/mu.json"),
    :ele=>FitResult("$BASE/results/fits/$(ARGS[4])/nominal/ele.json"),
    :combined=>FitResult("$BASE/results/fits/$(ARGS[4])/nominal/combined.json")
}
const bdt = "etajprime_topmass_default"
const BIAS = {:mu=>0.0183, :ele=>0.0333, :combined=>0.0240}

#Load histograms
    histos = readdir(unfolded_histos)
    histos = filter(x->contains(x, string(lepton)), histos) |> collect
    histos = filter(x->contains(x, ".root"), histos) |> collect;
    histos_d = Dict()
    for h in histos
        s = join(split(h, "__")[2:end], "__")
        s = s[1:end-5]
        histos_d[symbol(s)] = load_hists_from_file(
            "$unfolded_histos/$h", name->name in toload
        );

        tf = TFile("$unfolded_histos/$h")
        histos_d[symbol(s)]["unfolded"] = load_with_errors(tf, "unfolded";error_type = :errors)
        Close(tf)
        #println(s)
    end

    tms = {
        k => load_hists_from_file("$hist_path/$bdt/$lepton/tmatrix_nocharge__gen_$(k).root") for k in [:ele, :mu, :tau]
    };
    comphep_proj = tms[:ele]["tm__comphep__nominal__proj_x"] + tms[:mu]["tm__comphep__nominal__proj_x"] + tms[:tau]["tm__comphep__nominal__proj_x"]
    comphep_proj = comphep_proj * FITRESULTS[lepton][:beta_signal]
    hists = load_hists_from_file("$hist_path/$bdt/$lepton/cos_theta_lj.root") |> remove_prefix;

    unf1 = histos_d[:nominal]
    #println(hcat(contents(unf1["unfolded"]), errors(unf1["unfolded"])))
    haskey(unf1, "unfolded") || error("unfolding was not successful")

    hd = reweight_hists_to_fitres(FITRESULTS[lepton], hists)
    const nrebin=4
    hd = {k=>rebin(v, 2:nrebin:nbins(v)-nrebin) for (k, v) in hd};

    systs = map(x->symbol(x), map(x->split("$x", "__")[1], collect(keys(histos_d))) |> unique);

    const Ngen = integral(unf1["gen"])
    norm_gen(h) = (Ngen/integral(h)) * h

#Determine statistical error by covariant sampling
function asymmetry_error(unfolded, errmat)
    covm = contents(errmat)[2:end-2, 2:end-2]
    #ROOT matrix may not be posdef
    #symmetrize
    covm = triu(covm) + transpose(triu(covm)) - diagm(diag(covm))
    if !isposdef(covm)
        covm = diagm(errors(unfolded)[2:end-2])
    end
    Astat = 0
    try
        normd = MvNormal(contents(unfolded)[2:end-2], covm);
        asyms = Float64[]
        for i=1:100000
            x = rand(normd);
            a = asymmetry(x)
            push!(asyms, a)
        end
        Astat = std(asyms)
    catch err
        warn("could not calculate stat. error: $err")
        warn(string(covm))
        Astat = NaN
    end
    return Astat
end
    Astat = asymmetry_error(unf1["unfolded"], unf1["error"])
    println("Astat=", Astat)

#ASYMMETRIES
    asymmetries = {:up=>Dict(), :down=>Dict()}
    A = 0
    unf_histos = Dict()
    nom = nothing
    for (k, v) in histos_d
        (haskey(v, "unfolded") && haskey(v, "gen")) || error("unfolding $k was not successful")
        s = split(string(k), "__")
        syst = symbol(s[1])

        A = asymmetry(unf1["unfolded"])
        nom = deepcopy(unf1["unfolded"])

        if !haskey(unf_histos, syst) && syst != :nominal
            unf_histos[syst] = Dict()
        end
        if syst==:nominal
            continue
        elseif length(s)==1 && syst in [:mcstat, :stat, :fiterror]
            println(s, " ", contents(v["error"])|>diag)
            a = asymmetry_error(v["unfolded"], v["error"])
            asymmetries[:up][syst] = A + a
            asymmetries[:down][syst] = A - a
            unf_histos[syst][:up] = deepcopy(v["unfolded"])
            unf_histos[syst][:down] = deepcopy(v["unfolded"])
        elseif length(s)==1 && syst in [:generator]
            unf_histos[syst][:up] = deepcopy(v["unfolded"])
            unf_histos[syst][:down] = deepcopy(v["unfolded"])
            a = asymmetry(v["unfolded"])
            asymmetries[:up][syst] = a
            asymmetries[:down][syst] = a
        elseif length(s)==2
            dir = symbol(s[2])
            asymmetries[dir][syst] = asymmetry(v["unfolded"])
            unf_histos[syst][dir] = deepcopy(v["unfolded"])
        end
    end

    for (k, v) in unf_histos
        ax = axes()
        barplot(ax, norm_gen(nom), "black", label="nominal A=$(asymmetry(nom))")
        barplot(ax, unf1["gen"], "green", label=string("generated A=", asymmetry(unf1["gen"])))
        barplot(ax, norm_gen(v[:up]), "red", label="up A=$(asymmetry(v[:up]))")
        barplot(ax, norm_gen(v[:down]), "blue", label="down A=$(asymmetry(v[:down]))")
        legend(loc="best")
        xlabel(VARS[:cos_theta_lj])
        title("Unfolded $lepton $k")
        svfg("$BASE/results/plots/unfolding/$(k)__$(lepton)")
    end

    asymdf = DataFrame(syst=collect(keys(asymmetries[:up])));
    asymdf[:up] = Float64[asymmetries[:up][s] for s in asymdf[:syst]];
    asymdf[:down] = Float64[asymmetries[:down][s] for s in asymdf[:syst]];

    asymdf[:dup] = abs(A .- asymdf[:up]);
    asymdf[:ddown] = abs(A .- asymdf[:down]);
    mkpath("$BASE/results/tables/")

#    tot_du = asymdf[:dup].^2 |> sum |> sqrt
#    tot_dd = asymdf[:ddown].^2 |> sum |> sqrt;
    da_bias = BIAS[lepton]
    push!(asymdf, [:unfolding_bias, A + da_bias, A - da_bias, da_bias, da_bias])
    push!(asymdf, [:statistical, A+Astat, A-Astat, Astat, Astat])
    asymdf[:d] = max(abs(asymdf[:dup]), abs(asymdf[:ddown]))
    Asyst = asymdf[1:end-2, :d].^2 |> sum |> sqrt
    push!(asymdf, [:systematic, A+Asyst, A-Asyst, Asyst, Asyst, Asyst])
    push!(asymdf, [:nominal, A, A, 0.0, 0.0, 0.0])
    asymdf[:rel_percent] = 100.0 * asymdf[:d] ./ A
    #Asyst = sqrt(tot_du^2+tot_dd^2)
    writetable("$BASE/results/tables/asymmetries_$(lepton).csv", asymdf)
    println("Asyst=", Asyst, " A=", asymmetry(nom))

#MISC
    const nb_gen = nbins(unf1["gen"]) - 2
    const nb_reco = nbins(unf1["unfolded"]) - 2

    rbins = map(x->round(x, 2), edges(unf1["gen"])[2:end-1])

    function covmat(h::Histogram)
        errs = errors(h);
        errs[ errs .== 0] = 1;
        errm = diagm(errs.^2);
        return errm
    end

    function chi2_value(h1::Histogram, h2::Histogram, cov=nothing)
        if cov == nothing
            cov = covmat(h1)
        end
        return first(transpose(contents(h2) - contents(h1)) * inv(cov) * (contents(h2) - contents(h1)))
    end

    const cov = covmat(rebin(hd["DATA"], 2:nrebin:nb_reco-nrebin-1))

### DATA/MC chi2 plot
    const mctot = hd["tchan"]+hd["ttjets"]+hd["wjets"] + hd["qcd"] + hd["dyjets"] + hd["schan"] + hd["twchan"] + hd["gjets"];
    hmc = to_root(mctot, "mc");
    hdata = to_root(hd["DATA"], "data");
    chi2val = Chi2Test(hdata, convert(Ptr{TH1}, hmc.p), "UW CHI2/NDF")

    ax = axes()
    barplot(ax, hd["DATA"], "black", label="data");
    barplot(ax, mctot, "red", label="nominal signal model");
    grid();
    legend(loc="best");
    xlabel(VARS[:cos_theta_lj], fontsize=16);
    title("Reconstructed data vs. model, \$\\chi^2/n = $(round(chi2val, 2))\$, $(titles[lepton])");
    xticks(rbins);
    ylabel("reconstructed events / bin");
    svfg("$BASE/results/plots/unfolding/data_mc_comparison_$(lepton)");


###Unfolded plot
    fig = figure(num=nothing, figsize=(8, 8), dpi=200, facecolor="w", edgecolor="k")

    ax = axes()

    eplot(ax, norm_gen(unf1["unfolded"]), color="black", label="unfolded data", ls="none", marker="o")
    N = contents(norm_gen(unf1["unfolded"]))
    barplot(ax, norm_gen(unf1["gen"]), "green", label="generated (POWHEG)", do_error=false, lw=2)
    barplot(ax, rebin(comphep_proj, 2:nrebin:27-nrebin), "red", label="generated (CompHEP)", ls="--", lw=2, do_error=false)
    chi2unfolded0 = Chi2Test(
        to_root(unf1["gen"]),
        convert(Ptr{TH1}, to_root(norm_gen(unf1["unfolded"])).p),
        "UW CHI2/NDF"
    )
    x = (contents(unf1["gen"]) - contents(norm_gen(unf1["unfolded"])))
    x = x[2:end-2]
    println("x = ", x)

    covm = contents(unf1["error"])[2:end-2, 2:end-2]
    println("covm = ", covm)
    chi2unfolded = first(0.5 * transpose(x) * inv(covm) * x)
    println("chi2 unfolded = $chi2unfolded $chi2unfolded0")

    variations = {:up=>Any[], :down=>Any[]}
    for d in [:up, :down]
        for k in systs
            k in [:nominal, :stat, :mcstat, :fiterror] && continue
            #println("$k $d")
            delta = contents(
                norm_gen(unf1["unfolded"]) -
        #        norm_gen(histos_d[symbol("$(k)__$(d)")]["unfolded"])
                norm_gen(unf_histos[k][d])
            )[1:end-1] |> abs;
            push!(variations[d], deepcopy(delta))
        end
    end

    tot_du = [x.^2 for x in variations[:up]]|>sum|>sqrt
    tot_dd = [x.^2 for x in variations[:down]]|>sum|>sqrt
    N = N[1:end-1]
    println(tot_du, tot_dd)
    #tot_dd = sum(variations[:down].^2)|>sqrt

    ax[:bar](
        lowedge(unf1["gen"].bin_edges),
        tot_du + tot_dd,
        widths(unf1["gen"].bin_edges),
        N - tot_dd,
        edgecolor="grey",
        color="grey",
        fill=false,
        linewidth=0,
        hatch="/////",
        label="uncertainty";
    )

    #title("unfolded distribution, \$\\chi^2/n = $(round(chi2val, 2))\$")
    #grid()
    leg = legend(loc=4, numpoints=1)
    leg[:draw_frame](false)
    xlabel(string("", VARS[:cos_theta_lj]), fontsize=18)
    ylabel(VARS[:differential_cos_theta], fontsize=18)
    cmspaper_title(ax, 0.5, 1.01, additional_text=titles[lepton])
    unfolded = unf1["unfolded"]
    gen = unf1["gen"]
    text(0.5, 0.98,
        "A = $(round(asymmetry(unfolded), 2)) \$\\pm\$ $(round(Astat,2)) (stat.) \$\\pm\$ $(round(Asyst,2)) (syst.)\n",
        transform=ax[:transAxes], verticalalignment="top" , horizontalalignment="center",
        fontsize=16
    )

    @pyimport matplotlib.ticker as ticker

    majorLocator   = ticker.FixedLocator(linspace(-1, 1, length(rbins)))
    minorLocator   = ticker.FixedLocator(linspace(-1, 1, (length(rbins)-1)*4 + 1))

    ax[:xaxis][:set_major_locator](majorLocator)
    ax[:xaxis][:set_minor_locator](minorLocator)
    ax[:ticklabel_format](axis="y", style="sci", useOffset=true, scilimits=(-2, 2))
    ax[:yaxis][:major][:formatter][:_useMathText] = true
    #rcParams["text.usetex"] = true

    #xticks(rbins);
    ax[:tick_params](axis="x",which="minor",bottom="on")
    svfg("$BASE/results/plots/unfolded_$(lepton)");

# tot_tm = unf1["genrec"];
# effh = zeros(nb_gen)
# for i=1:nb_gen-1
#     a = contents(tot_tm)[i+1, :]
#     effh[i] = sum(a[2:end]/sum(a))
#     if !(effh[i] > 0)
#         effh[i] = 0
#     end
# end
# effh[nb_gen] = 0;
# effh2 = zeros(nb_reco)
# for i=1:nb_gen-1
#     effh2[2*i-1] = effh[i]/2
#     effh2[2*i] = effh[i]/2
# end
# effh2[nb_reco] = 0
#
# tm = transpose(contents(tot_tm)[2:end-1, 2:end-1]);
# for i=1:nb_gen
#     tm[:, i] = tm[:, i]/sum(tm[:, i])
# end
# tm[isnan(tm)] = 0
# tm
#
# x = tm * (effh .* contents(unf1["unfolded"])[2:end-1]);
# h = Histogram([10000000 for i=1:nb_reco], x, linspace(-1, 1, nb_reco));
# ax = axes()
# barplot(ax, normed(h), "red", label="unfolded \$\\times\$ efficiency \$\\times\$ migration")
# barplot(ax, normed(hd["tchan"]), "green", label="reconstructed signal")
# legend(loc=2)
# xlabel(VARS[:cos_theta_lj], fontsize=16)
# ylabel("reconstructed events")
# grid()
# xticks(rbins)
# svfg("$BASE/results/plots/unfolding/unfolded_comparison_$(lepton)")
#
# hratio = Histogram(
# [1 for i=1:nb_reco+2],
# vcat(0, contents(h), 0) ./ contents(hd["tchan"]),
#     edges(hd["tchan"])
# );
# hratio.bin_contents[isnan(hratio.bin_contents)] = 0
#
# hratio_n = Histogram(
# [1 for i=1:nb_reco+2],
# vcat(0, contents(normed(h)), 0) ./ contents(normed(hd["tchan"])),
#     edges(hd["tchan"])
# );
# hratio_n.bin_contents[isnan(hratio_n.bin_contents)] = 0
#
#
# # In[197]:
#
# ax = axes()
# barplot(ax, hratio, "black"; do_error=false, ls="-.", label="unfolded / nominal")
# barplot(ax, hratio_n, "black"; do_error=false, ls="-", label="unfolded / nominal (normed)")
# axhline(1.0, label="nominal", color="black", ls="--")
# grid()
# ylim(0.6, 1.6)
# xlabel(VARS[:cos_theta_lj], fontsize=16)
# legend(loc="best")
# xticks(rbins)
# svfg("$BASE/results/plots/unfolding/unfolded_comparison_ratio_$(lepton)")
#
#

#Reco-level plot
hd = reweight_hists_to_fitres(FITRESULTS[lepton], hists)
hd = {k=>rebin(v, 2:nrebin:nbins(v)-nrebin-1) for (k, v) in hd};
(ax, rax) = combdraw(hd, :cos_theta_lj; loc_paperstring=(:top, :left),
titletext=titles[lepton]
);

hbg = sum([hd[k] for k in ["wjets", "ttjets", "qcd", "schan", "twchan", "dyjets", "gjets"]])
#htchan = Histogram(entries(hbg), contents(hbg) + vcat(0, contents(h), 0), edges(hd["DATA"]))
#htchan = rebin(htchan, 2:nrebin:nbins(htchan)-nrebin-1)
#htchan = integral(hd["DATA"])/integral(htchan) * htchan
#if lepton == :mu
#    htch_asym = hd["tchan__asym_028"] + hbg
#elseif lepton == :ele
#    htch_asym = hd["tchan__asym_008"] + hbg
#end

#barplot(ax, htchan, "black", label="unfolded", do_error=false)
#barplot(rax, hd["DATA"]/htchan, "black", do_error=false)
#barplot(ax, htch_asym, "purple", label="weighted to A", do_error=false)
#barplot(rax, (hd["DATA"] - htch_asym)/hd["DATA"], "purple", do_error=false)

ylabel(VARS[:differential_cos_theta], fontsize=18)
rslegend(ax)
xticks(rbins)
svfg("$BASE/results/plots/results/cos_theta_lj_2j_1t_$(lepton)_bdt_0_6")
#
#
# #### Correlation matrix parametrization
# hcorr = unf1["correlation"];
#
# imshow(contents(hcorr)[2:end-2, 2:end-2], interpolation="nearest")
# colorbar()
# xlabel("bins")
# ylabel("bins")
# title("bin-by-bin correlation")
# svfg("$BASE/results/plots/unfolding/unfolded_bin_correlation_$(lepton)")

#
#
## In[202]:
#
## eigs = eigvals(covm);
## ndf = 0
## for i=1:length(eigs)
##     ndf += eigs[i]/(eigs[i] + 1E-5)
## end
## ndf
#
#
## In[203]:
#
## asym(x) = -(sum(x[1:length(x)/2]) - sum(x[length(x)/2+1:end])) / (sum(x[1:length(x)/2]) + sum(x[length(x)/2+1:end]))
#
#
##### Biascheck of asymmetry calculation
#
## In[204]:
#
## fpoly(x, vs...) = sum([vs[i] * x.^(i-1) for i=1:length(vs)])
#
#
## In[205]:
#
## function linear_fit(x, yv, covm)
#
##     xv = x
##     function myfunc(x::Vector{Float64}, grad::Vector{Float64})
##         z = first(transpose(fpoly(xv, x...) - yv) * inv(covm[2:end-2, 2:end-2]) * (fpoly(xv, x...) - yv))
##         return z
##     end
#
##     priors = [25000, 25000];
#
##     opt = Opt(:LN_COBYLA, length(priors))
##     #lower_bounds!(opt, [0, 0])
##     #upper_bounds!(opt, [10.0, 10.0])
##     xtol_rel!(opt,1e-8)
#
##     min_objective!(opt, myfunc)
#
##     (minf,minx,ret) = optimize(opt, priors)
##     #println("got $minf at $minx after $count iterations (returned $ret)")
##     return minx
## end
#
#
## In[206]:
#
## x = edges(unf1["unfolded"])[2:end-2];
#
## ydata = contents(unf1["unfolded"])[2:end-2];
## yref = contents(unf1["gen"])[2:end-2]
## minx = linear_fit(x, ydata, covm)
#
## asym1 = asymmetry(unf1["unfolded"])
## asym2 = asymmetry(unf1["gen"])
#
## asym3 = fpoly(x, minx...) |> asym
#
## errorbar(midpoints(rbins), ydata, stds[2:end-2], lw=1, color="k", marker="o", label="unfolded A=$(round(asym1, 2))", ls="none")
## plot(midpoints(rbins), yref, lw=0, marker="o", label="generated A=$(round(asym2, 2))")
## plot(-1:0.01:1, fpoly(-1:0.01:1, minx...), label="linear fit A=$(round(asym3, 2))\n [\$a_0,a_1\$]=$(map(x->int(round(x,0)), minx))")
## xlim(-1.1, 1.1)
## xticks(rbins)
## grid()
## xlabel(VARS[:cos_theta_lj], fontsize=16)
## legend(loc="best", numpoints=1)
## svfg("$BASE/results/plots/$ts/unfolding/linear_fit_$(lepton)")
#
#
## In[207]:
#
## for i=1:200
##     plot(linspace(-1, 1, nb_gen-1), rand(nd, 1), alpha=0.2, color="gray")
## end
## xlim(-1, 1)
## #ylim(0, 90000)
## grid()
## xlabel(VARS[:cos_theta_lj], fontsize=16)
## ylabel("unfolded event yield")
## title("statistical variation of unfolded distribution")
#
#
## In[208]:
#
## N = 10000
## asyms = zeros(N)
## asyms2 = zeros(N)
## asyms3 = zeros(N)
## for i=1:N
##     s = rand(nd, 1)
##     _a = asym(s[:])
##     asyms[i] = _a
#
##     #with all bins
##     minx2 = linear_fit(x, s[:], covm)
##     _asym2 = fpoly(x, minx2...) |> asym
##     asyms2[i] = _asym2
#
##     #without last bin
##     minx3 = linear_fit(x[1:end-1], s[:][1:end-1], covm[1:end-1,1:end-1])
##     _asym3 = fpoly(x, minx3...) |> asym
##     asyms3[i] = _asym3
#
#
## end
#
#
## In[209]:
#
## PyPlot.plt.hist(asyms, bins=25, lw=0, alpha=0.5, normed=true,
## label="sum stat. spread \n\$\\mu=$(round(mean(asyms), 2)),\\ \\sigma=$(round(std(asyms), 2))\$")
## PyPlot.plt.hist(asyms2, bins=25, lw=0, alpha=0.5, normed=true,
## label="$(nb_gen - 1)-bin fit stat. spread\n\$\\mu=$(round(mean(asyms2), 2)),\\ \\sigma=$(round(std(asyms2), 2))\$")
#
## PyPlot.plt.hist(asyms3, bins=25, lw=0, alpha=0.5, normed=true,
## label="$(nb_gen - 2)-bin fit stat. spread\n\$\\mu=$(round(mean(asyms3), 2)),\\ \\sigma=$(round(std(asyms3), 2))\$")
#
#
## axvline(asym(ydata), color="black", label="summed value\n\$A=$(round(asym1, 2))\$")
## axvline(asym3, color="green", label="fitted value\n\$A=$(round(asym3, 2))\$")
## grid()
## xlabel("asymmetry", fontsize=16)
## leg = legend(loc="best")
## leg[:get_frame]()[:set_alpha](0.5)
## #ylim(top=50000)
## title("statistical uncertainty of the unfolded asymmetry")
## svfg("$BASE/results/plots/$ts/unfolding/asymmetry_sum_fit_$(lepton)")
#
#
#### Transfer matrix parametrizations
#
## In[210]:
#
#function gaussfit(h::Histogram)
#    f2(x, mu, sigma) = pdf(Normal(mu, sigma), x)
#
#    const xv = midpoints(edges(h))
#    const yv = contents(h)[1:end-1]
#    const covm = covmat(h)[1:end-1, 1:end-1]
#
#    function myfunc(x::Vector{Float64}, grad::Vector{Float64})
#        z = first(transpose(f2(xv, x...) - yv) * inv(covm) * (f2(xv, x...) - yv))
#        return z
#    end
#
#    priors = [0, 1];
#
#    opt = Opt(:LN_COBYLA, length(priors))
#    lower_bounds!(opt, [-5, 0.1])
#    upper_bounds!(opt, [+5, 1])
#    xtol_rel!(opt,1e-25)
#
#    min_objective!(opt, myfunc)
#
#    (minf,minx,ret) = optimize(opt, priors)
#    #println("got $minf at $minx after $count iterations (returned $ret)")
#    return (Normal(minx...), minf)
#end
#
#
## In[211]:
#
#contents(tot_tm)
#
#
## In[212]:
#
#df = similar(DataFrame(
#x=Float64[], mp=Float64[], mean=Float64[], std=Float64[], gof=Float64[], nreco=Float64[], nreco_good=Float64[], hist=Histogram[]), nb_gen-1)
#for i=1:nb_gen-1
#    h = Histogram(
#        entries(tot_tm)[1+i, 2:end-1][:],
#        contents(tot_tm)[1+i, 2:end-1][:],
#        linspace(-1, 1, nb_reco)
#    );
#    (g, l) = gaussfit(h)
#    #println(g)
#    #println(mean(g), " ", std(g), " ", l)
#    df[i, :x] = linspace(-1, 1, nb_gen)[i]
#    df[i, :mp] =  midpoints(rbins)[i]
#
#    df[i, :nreco] = round(sum(contents(h)))
#    df[i, :nreco_good] = round(sum(contents(h)[(2 * i - 1) : (2 * i)]))
#
#    df[i, :mean] = round(mean(g), 2)
#    df[i, :std] = round(std(g), 2)
#    df[i, :gof] = l
#    df[i, :hist] = h
#
#end
#df[[:mp, :mean, :std, :nreco, :nreco_good]]
#
#
## In[213]:
#
#errorbar(rbins|>midpoints, df[:mean][1:end], df[:std][1:end], label="gaussian fit")
#plot(-1:1, -1:1, label="identity migration")
##ylim(-0.5, 2)
#xticks(rbins, rotation=90)
##xlim(linspace(-1, 1, nb_gen)[13], linspace(-1, 1, nb_gen)[end])
#grid()
#legend(loc=2)
#xlabel(VARS[:cos_theta_lj_gen], fontsize=16)
#ylabel(string("reconstructed ", VARS[:cos_theta_lj]), fontsize=16)
#
#
## In[214]:
#
#function plot_bin_migration(n)
#    ax = axes()
#    ax[:plot](-1:0.01:1, pdf(Normal(df[n, :mean], df[n, :std]), -1:0.01:1)*(2/nb_reco),
#    label="normal fit, \$\\mu = $(round(df[n, :mean], 2))\$, \$\\sigma = $(round(df[n, :std], 2))\$"
#    )
#    ncorrect = sum(contents(df[n, :hist])[(2 * n - 1) : (2 * n)])
#
#    barplot(ax, df[n, :hist]|>normed, "red",
#    label="reconstructed values\nN=$(int(sum(contents(df[n, :hist])))), N($n)=$(int(ncorrect))"
#    )
#    axvline(rbins[n], color="black", ls="--", label="generated bin")
#    axvline(rbins[n+1], color="black", ls="--")
#
#    contents(df[n, :hist])[n]
#    leg = legend(loc="best")
#    leg[:get_frame]()[:set_alpha](0.5)
#    xticks(rbins, rotation=90)
#    grid()
#    title("reconstruction bias for bin $n, $(VARS[:cos_theta_lj]) in [$(rbins[n]), $(rbins[n+1]))")
#end
#
#
## In[215]:
#
#plot_bin_migration(1)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_1_$(lepton)")
#
#
## In[216]:
#
#plot_bin_migration(2)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_2_$(lepton)")
#
#
## In[217]:
#
#plot_bin_migration(3)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_3_$(lepton)")
#
#
## In[218]:
#
#plot_bin_migration(4)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_4_$(lepton)")
#
#
## In[219]:
#
#plot_bin_migration(5)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_5_$(lepton)")
#
#
## In[220]:
#
#plot_bin_migration(6)
#svfg("$BASE/results/plots/$ts/unfolding/tm_bin_6_$(lepton)")
#
#
## In[221]:
#
#asyms = Any[]
#for x in ["-0.20000", "-0.10000", "0.00000", "0.10000", "0.20000", "0.30000", "0.40000", "0.50000", "0.60000", "0.70000"]
#    unf1 = load_hists_from_file("$BASE/src/stpol-unfold/results/csvt_qcd_mva__bdt_$(x)__$(lepton).root", name->name in toload);
#    asym = unf1["unfolded"]|>asymmetry
#    push!(asyms, asym)
#end
#
##plot(asyms)
