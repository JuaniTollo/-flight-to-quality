# ============================================================
# models.jl — osciladores en Julia (solver + PINN inversa)
# ============================================================
# Capa de MODELOS del experimento (la evaluación vive en Python).
# Para cada crisis (2001/2008/2020): entrena FN y LV sobre la ventana de ciclos previos
# (lookback) por DOS métodos —solver single-shooting (restricción dura) y PINN inversa
# (restricción blanda, eje λ)— y pronostica el episodio de crisis con origen móvil.
# Exporta output/experiment/julia_forecasts.csv (crisis,model,method,h,target_date,
# series,y_true,y_pred) para que Python lo compare ALINEADO con RW/VAR.
#
# Corre:  julia --project=julia \
#               src/experiment/models.jl
# ============================================================
using DifferentialEquations, Optim, Lux, Optimisers, Zygote, Random, Statistics
using DelimitedFiles, Dates, Printf
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const REPO   = normpath(joinpath(@__DIR__, "..", ".."))
const CSVP   = joinpath(REPO, "data", "processed", "lotka_volterra.csv")
const OUTDIR = joinpath(REPO, "output", "experiment"); mkpath(OUTDIR)

const HORIZONS = [1, 2, 4]
const LOOKBACK = 12.0          # años de entrenamiento previos a la crisis
# crisis -> (pico, valle)
const CRISES = [("2001", "2001-03-31", "2001-12-31"),
                ("2008", "2007-12-31", "2009-06-30"),
                ("2020", "2020-03-31", "2020-06-30")]

# --------------------------------------------------------------- datos
function load_yoy(path)
    raw, h = readdlm(path, ',', header=true); hv = vec(h)
    di = findfirst(==("DATE"), hv); xi = findfirst(==("PROFITS_YOY"), hv); yi = findfirst(==("INVEST_YOY"), hv)
    return Date.(string.(raw[:, di])), Float64.(raw[:, xi]), Float64.(raw[:, yi])
end

# --------------------------------------------------------------- modelos físicos
fn_rhs(u, p) = (p[3] * (u[1] - u[1]^3 / 3 - u[2] + p[4]), (u[1] - p[1] - p[2] * u[2]) / p[3])  # a,b,c,I
lv_rhs(u, p) = (p[1] * u[1] - p[2] * u[1] * u[2], p[3] * u[1] * u[2] - p[4] * u[2])             # α,β,δ,γ

struct Osc
    name::String
    rhs::Function
    p0::Vector{Float64}
    needs_offset::Bool
    guard::Function
end
const FN = Osc("FN", fn_rhs, [0.2, 0.3, 3.0, 0.0], false, p -> p[3] > 0)
const LV = Osc("LV", lv_rhs, [1.0, 0.3, 0.4, 1.5], true, p -> all(p .> 0))

function osolve(m::Osc, θ, u0, tspan, saveat)
    (all(isfinite, θ) && m.guard(θ)) || return nothing
    f!(du, u, p, t) = (r = m.rhs(u, p); du[1] = r[1]; du[2] = r[2])
    sol = solve(ODEProblem(f!, u0, tspan, θ), Tsit5(); saveat=saveat, abstol=1e-6, reltol=1e-6, maxiters=10^5)
    (sol.retcode == ReturnCode.Success && length(sol.t) == length(saveat)) || return nothing
    return Array(sol)
end

# --------------------------------------------------------------- método A: solver
function fit_solver(m::Osc, t, V, R)
    u0 = [V[1], R[1]]; uobs = permutedims(hcat(V, R))
    function loss(θ)
        y = osolve(m, θ, u0, (t[1], t[end]), t)
        y === nothing && return 1e8
        return sum((y .- uobs) .^ 2)
    end
    best, bf = copy(m.p0), Inf
    for s in (1.0, 1.2, 0.8)
        res = optimize(loss, m.p0 .* s, NelderMead(), Optim.Options(iterations=2000))
        Optim.minimum(res) < bf && (bf = Optim.minimum(res); best = Optim.minimizer(res))
    end
    return best
end

# --------------------------------------------------------------- método B: PINN inversa
function build_net(Tyears, oscale, seed)
    nn = Chain(WrappedFunction(t -> t ./ Tyears),
               Dense(1 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 2),
               WrappedFunction(x -> oscale .* x))
    ps, st = Lux.setup(MersenneTwister(seed), nn)
    return nn, Lux.f64(ps), Lux.testmode(st)
end
nn_dudt(nn, t, ps, st; h=1e-4) = (nn([t + h], ps, st)[1] .- nn([t - h], ps, st)[1]) ./ (2h)

function fit_pinn(m::Osc, t, V, R; λ=1.0, epochs=2500, lr=0.01, ncol=120, seed=42)
    Tyears = t[end] - t[1]; oscale = maximum(abs, vcat(V, R)) + 1e-6
    nn, ps, st = build_net(Tyears, oscale, seed)
    uobs = [[V[j], R[j]] for j in eachindex(t)]
    theta = (nn=ps, ode=copy(m.p0))
    opt = Optimisers.setup(Optimisers.Adam(lr), theta)
    rng = MersenneTwister(seed + 1)
    tcol = rand(rng, ncol) .* Tyears .+ t[1]
    for epoch in 1:epochs
        epoch % 50 == 0 && (tcol = rand(rng, ncol) .* Tyears .+ t[1])
        _, g = Zygote.withgradient(theta) do th
            dl = mean(sum((nn([t[j]], th.nn, st)[1] .- uobs[j]) .^ 2) for j in eachindex(t))
            pl = mean(tcol) do ti
                u = nn([ti], th.nn, st)[1]; du = nn_dudt(nn, ti, th.nn, st)
                r = m.rhs(u, th.ode)
                (du[1] - r[1])^2 + (du[2] - r[2])^2
            end
            dl + λ * pl
        end
        opt, theta = Optimisers.update(opt, theta, g[1])
    end
    return theta.ode
end

# --------------------------------------------------------------- forecast origen-móvil
# Devuelve filas (method, h, target_date, prof_pred, inv_pred, prof_true, inv_true) en YoY.
function forecast_rows(m::Osc, method::String, θ, dates, t, idx_episode, V, R, to_raw)
    rows = NamedTuple[]
    n = length(t)
    for ti in idx_episode
        for h in HORIZONS
            oi = ti - h
            oi < 1 && continue
            hy = t[ti] - t[oi]
            y = osolve(m, θ, [V[oi], R[oi]], (0.0, hy), [hy])
            y === nothing && continue
            pp = to_raw(y[1, 1], 1); ip = to_raw(y[2, 1], 2)
            push!(rows, (method=method, h=h, target=dates[ti],
                         pp=pp, ip=ip, pt=to_raw(V[ti], 1), it=to_raw(R[ti], 2)))
        end
    end
    return rows
end

# --------------------------------------------------------------- main
function main()
    dates, prof, inv = load_yoy(CSVP)
    tof(d, d0) = Dates.value(d - d0) / 365.25
    open(joinpath(OUTDIR, "julia_forecasts.csv"), "w") do io
        println(io, "crisis,model,method,h,target_date,series,y_true,y_pred")
        for (cl, pk, tr) in CRISES
            peak = Date(pk); trough = Date(tr)
            w_lo = peak - Month(6); w_hi = trough + Month(12)
            tr_mask = (dates .>= w_lo - Year(Int(LOOKBACK))) .& (dates .< w_lo)
            xt, yt = prof[tr_mask], inv[tr_mask]
            μx, σx = mean(xt), std(xt); μy, σy = mean(yt), std(yt)
            for m in (FN, LV)
                offset = 0.0
                if m.needs_offset
                    zx = (xt .- μx) ./ σx; zy = (yt .- μy) ./ σy
                    offset = -min(minimum(zx), minimum(zy)) + 1.0
                end
                to_z(v, k) = k == 1 ? (v - μx) / σx + offset : (v - μy) / σy + offset
                to_raw(z, k) = k == 1 ? (z - offset) * σx + μx : (z - offset) * σy + μy

                # serie completa en z (para anclar estado en cualquier origen)
                d0 = dates[findfirst(tr_mask)]
                t_all = [tof(d, d0) for d in dates]
                V = [to_z(prof[i], 1) for i in eachindex(dates)]
                R = [to_z(inv[i], 2) for i in eachindex(dates)]

                # entrenamiento (ventana lookback)
                idx_tr = findall(tr_mask)
                tt = t_all[idx_tr]; Vt = V[idx_tr]; Rt = R[idx_tr]
                tt = tt .- tt[1]                       # 0-based para el fit

                θ_solver = fit_solver(m, tt, Vt, Rt)
                θ_pinn   = fit_pinn(m, tt, Vt, Rt)

                idx_ep = findall((dates .>= w_lo) .& (dates .<= w_hi))
                for (method, θ) in (("solver", θ_solver), ("pinn", θ_pinn))
                    rows = forecast_rows(m, method, θ, dates, t_all, idx_ep, V, R, to_raw)
                    for r in rows
                        @printf(io, "%s,%s,%s,%d,%s,profits,%.4f,%.4f\n", cl, m.name, r.method, r.h, r.target, r.pt, r.pp)
                        @printf(io, "%s,%s,%s,%d,%s,invest,%.4f,%.4f\n",  cl, m.name, r.method, r.h, r.target, r.it, r.ip)
                    end
                    @printf("✓ %s %-2s %-6s: %d pronósticos | θ=[%s]\n",
                            cl, m.name, method, length(rows), join((@sprintf("%.3f", v) for v in θ), ","))
                end
            end
        end
    end
    println("✓ $(joinpath(OUTDIR, "julia_forecasts.csv"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
