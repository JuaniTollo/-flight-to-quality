# ============================================================
# p12_pinn_profile.jl — PERFIL DE VEROSIMILITUD del lag de maduración (μ)
# ============================================================
# Mismo modelo físico que p11_pinn_maturation.jl (cadena de maduración profit→investment,
# kernel Gamma de media μ, ODE diferenciable calibrada por PINN inverso):
#
#   m(t) = Σⱼ wⱼ(μ) · I(t − j)          (inversión madurada; wⱼ = kernel Gamma, media μ)
#   dP/dt = c·I − d·P − k·m              (ganancia: + demanda actual, − sobreacumulación madurada)
#   dI/dt = a·P − b·I                    (acelerador)
#
# PROBLEMA QUE RESUELVE: en p11, optimizar μ JUNTO con (k,…) deja un tradeoff residual k–μ
# que sesga el lag recuperado (sintético 4 trim → ~1.86). PERFIL DE VEROSIMILITUD: en vez de
# estimar μ libremente, se FIJA μ en una grilla de valores; para cada μ fijo se ajusta TODO el
# resto (red + a,b,c,d,k), y se registra la pérdida mínima alcanzada. La curva perfil(μ) =
# min_resto L(μ) rompe el tradeoff (k ya no puede compensar a μ porque μ no se mueve) y su
# mínimo da el μ̂; la curvatura/ancho alrededor del mínimo da un IC aproximado.
#
# IC aproximado: por la teoría de la verosimilitud perfilada, ΔL = L(μ) − L(μ̂) define un
# intervalo {μ : ΔL ≤ umbral}. Acá L es un MSE (no log-verosimilitud), así que el IC se reporta
# como el rango donde el perfil sube menos de un factor relativo sobre el mínimo (heurístico,
# documentado abajo) — sirve como ancho de identificabilidad, no como cobertura frecuentista.
#
# Transform: QoQ (pct_change(1)), el insesgado (ver §4 de notes/STATUS.md). Las series del CSV
# vienen en niveles (PROFITS, INVESTMENT) → se computa QoQ acá, NO se usan las columnas *_YOY.
#
# Parte A: VALIDACIÓN EN SINTÉTICO — el mínimo del perfil debe caer en el lag conocido (4 trim).
# Parte B: DATOS REALES — perfil del lag de maduración sobre una ventana representativa.
#
# Corre (grilla completa):  julia --project=julia src/experiment/p12_pinn_profile.jl
# Smoke test (rápido):      julia --project=julia src/experiment/p12_pinn_profile.jl --smoke
# ============================================================
using DifferentialEquations, Lux, Optimisers, Zygote, Random, Statistics, Printf, DelimitedFiles

const SHAPE  = 4.0          # forma del kernel Gamma (fija; el ancho no es identificable, sí la media μ)
const KMAX   = 10           # rezagos del kernel (trimestres)
const λ_phys = 1.0
const REPO   = abspath(joinpath(@__DIR__, "..", ".."))
const CSVP   = joinpath(REPO, "data", "processed", "lotka_volterra.csv")
const RESDIR = joinpath(REPO, "results"); mkpath(RESDIR)

# Grilla de μ para el perfil (trimestres). En smoke test se usa un subconjunto (ver main()).
const MU_GRID_FULL  = collect(1.0:1.0:10.0)
const MU_GRID_SMOKE = [2.0, 4.0, 6.0]

# ---------------------------------------------------------------- kernel Gamma (μ FIJO en el perfil)
function gamma_kernel(μ)                      # media μ trimestres, forma SHAPE
    scale = μ / SHAPE
    xs = collect(0:KMAX) .+ 0.5
    w = (xs .^ (SHAPE - 1)) .* exp.(-xs ./ scale)
    return w ./ sum(w)
end

# ---------------------------------------------------------------- generación sintética (cadena real)
function chain_rhs!(du, u, p, t)
    P, I, m1, m2, m3, m4 = u
    a, b, c, d, k, θ = p
    du[1] = c*I - d*P - k*m4
    du[2] = a*P - b*I
    du[3] = (I  - m1)/θ; du[4] = (m1 - m2)/θ; du[5] = (m2 - m3)/θ; du[6] = (m3 - m4)/θ
end

# ---------------------------------------------------------------- red
function build_nn(t0, T, oscale)
    Chain(WrappedFunction(t -> (t .- t0) ./ (T - t0)),
          Dense(1=>32, tanh), Dense(32=>32, tanh), Dense(32=>16, tanh), Dense(16=>2),
          WrappedFunction(x -> oscale .* x))
end
nn_dudt(nn, t, ps, st; h=1e-4) = (nn([t+h], ps, st)[1] .- nn([t-h], ps, st)[1]) ./ (2h)

# ---------------------------------------------------------------- entrenamiento con μ FIJO
# A diferencia de p11, μ NO es entrenable: el kernel w se precomputa de μ_fix y entra como
# constante. Sólo se optimizan los pesos de la red y p̂ = [a,b,c,d,k]. Devuelve la pérdida
# mínima alcanzada (componentes data/phys) y los params físicos ajustados.
function train_fixed_mu(t, PI_obs, p_init, oscale, μ_fix; n_epochs=8000, seed=42, lr=1e-2, verbose=false)
    n = length(t)
    w = gamma_kernel(μ_fix)                  # CONSTANTE: μ fijo rompe el tradeoff k–μ
    nn = build_nn(t[1], t[end], oscale)
    nn_ps, st = Lux.setup(MersenneTwister(seed), nn); st = Lux.testmode(st)
    nn_ps = Lux.f64(nn_ps)
    theta = (nn = nn_ps, ode = copy(p_init))   # ode = [a,b,c,d,k] (5 params, SIN logμ)

    function comps(th)
        ps = th.nn
        a,b,c,d,k = th.ode[1],th.ode[2],th.ode[3],th.ode[4],th.ode[5]
        P = [nn([t[i]], ps, st)[1][1] for i in 1:n]
        I = [nn([t[i]], ps, st)[1][2] for i in 1:n]
        data = mean((P .- PI_obs[1,:]).^2 .+ (I .- PI_obs[2,:]).^2)
        m = [sum(w[j+1]*I[i-j] for j in 0:min(KMAX, i-1)) for i in 1:n]
        phys = mean(map((KMAX+1):(n-1)) do i
            du = nn_dudt(nn, t[i], ps, st)
            rP = du[1] - (c*I[i] - d*P[i] - k*m[i])
            rI = du[2] - (a*P[i] - b*I[i])
            rP^2 + rI^2
        end)
        return data, λ_phys*phys
    end
    loss(th) = sum(comps(th))

    opt = Optimisers.setup(Adam(lr), theta)
    for epoch in 1:n_epochs
        _, g = Zygote.withgradient(loss, theta)
        opt, theta = Optimisers.update(opt, theta, g[1])
        if verbose && epoch % max(1, n_epochs ÷ 4) == 0
            dl, pl = comps(theta)
            @printf("    [μ=%.1f] época %5d │ L_data=%.3e │ L_phys=%.3e │ k=%.3f\n",
                    μ_fix, epoch, dl, pl, theta.ode[5])
        end
    end
    dl, pl = comps(theta)
    return (loss=dl+pl, data=dl, phys=pl, ode=theta.ode, nn=theta.nn, st=st)
end

# ---------------------------------------------------------------- barrido del perfil
# Para cada μ en la grilla, ajusta el resto con μ fijo y registra la pérdida mínima.
# Devuelve (mus, losses, fits) con el ajuste completo por μ (para inspección).
function profile_mu(t, PI_obs, p_init, oscale, mu_grid; n_epochs=8000, verbose=true)
    losses = Float64[]; fits = Any[]
    for μ in mu_grid
        fit = train_fixed_mu(t, PI_obs, p_init, oscale, μ; n_epochs=n_epochs, verbose=verbose)
        push!(losses, fit.loss); push!(fits, fit)
        verbose && @printf("  μ=%5.2f trim │ L=%.4e (data=%.3e, phys=%.3e) │ k̂=%.3f\n",
                           μ, fit.loss, fit.data, fit.phys, fit.ode[5])
    end
    return collect(mu_grid), losses, fits
end

# ---------------------------------------------------------------- IC aproximado del perfil
# Mínimo del perfil + rango de identificabilidad. Como L es un MSE, se usa un umbral relativo:
# {μ : L(μ) ≤ L_min·(1+tol)} con interpolación lineal a los bordes. Es una medida de ancho de
# identificabilidad (curvatura del perfil), NO un IC con cobertura frecuentista garantizada.
function profile_summary(mus, losses; tol=0.10)
    imin = argmin(losses)
    μ̂ = mus[imin]; Lmin = losses[imin]
    thr = Lmin * (1 + tol)
    # borde inferior
    lo = mus[1]
    for i in imin:-1:2
        if losses[i-1] > thr
            frac = (thr - losses[i]) / (losses[i-1] - losses[i])
            lo = mus[i] + frac*(mus[i-1] - mus[i]); break
        end
        lo = mus[i-1]
    end
    # borde superior
    hi = mus[end]
    for i in imin:(length(mus)-1)
        if losses[i+1] > thr
            frac = (thr - losses[i]) / (losses[i+1] - losses[i])
            hi = mus[i] + frac*(mus[i+1] - mus[i]); break
        end
        hi = mus[i+1]
    end
    return μ̂, Lmin, lo, hi
end

# ---------------------------------------------------------------- Parte A: sintético
function part_A(mu_grid; n_epochs)
    println("="^72); println("PARTE A — VALIDACIÓN EN SINTÉTICO (perfil debe minimizar en el lag conocido)"); println("="^72)
    θ_true = 0.25                                   # años/etapa → lag = 4·0.25·4 = 4 trim = 1 año
    μ_true_q = 4.0                                  # lag verdadero en trimestres (1 año)
    p_chain = [0.9, 0.4, 0.6, 0.4, 1.1, θ_true]
    u0 = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]; tspan = (0.0, 24.0)
    sol = solve(ODEProblem(chain_rhs!, u0, tspan, p_chain), Tsit5(), saveat=0.25, abstol=1e-8, reltol=1e-8)
    t = collect(0.0:0.25:24.0)
    PI = hcat([sol(τ)[1:2] for τ in t]...)
    PI .+= 0.03*std(PI).*randn(MersenneTwister(0), size(PI))
    oscale = maximum(abs.(PI))
    p_init = [0.5, 0.2, 0.3, 0.2, 0.5]              # init de [a,b,c,d,k]; μ lo barre la grilla
    @printf("verdadero: lag μ=%.1f trim (1 año), k=%.2f │ grilla μ = %s\n",
            μ_true_q, p_chain[5], string(mu_grid))
    mus, losses, _ = profile_mu(t, PI, p_init, oscale, mu_grid; n_epochs=n_epochs)
    μ̂, Lmin, lo, hi = profile_summary(mus, losses)
    @printf("\nPERFIL (sintético): mínimo en μ̂=%.2f trim (verdadero=%.2f) │ IC≈[%.2f, %.2f] │ L_min=%.4e\n",
            μ̂, μ_true_q, lo, hi, Lmin)
    return μ_true_q, mus, losses, μ̂, lo, hi
end

# ---------------------------------------------------------------- carga datos reales (QoQ desde niveles)
function load_window_qoq(; from_year=1990)
    raw, h = readdlm(CSVP, ',', header=true); hv = vec(h)
    di = findfirst(==("DATE"), hv)
    pi_ = findfirst(==("PROFITS"), hv); ii = findfirst(==("INVESTMENT"), hv)
    rows = [r for r in 1:size(raw,1) if raw[r,pi_] != "" && raw[r,ii] != ""]
    dts  = [string(raw[r,di]) for r in rows]
    Plv  = Float64[raw[r,pi_] for r in rows]; Ilv = Float64[raw[r,ii] for r in rows]
    # QoQ = pct_change(1) sobre niveles (el transform insesgado; ver §4 de STATUS.md)
    Pq = diff(Plv) ./ Plv[1:end-1] .* 100
    Iq = diff(Ilv) ./ Ilv[1:end-1] .* 100
    dq = dts[2:end]
    keep = [parse(Int, d[1:4]) >= from_year for d in dq]
    z(v) = (v .- mean(v)) ./ std(v)
    return z(Pq[keep]), z(Iq[keep])
end

# ---------------------------------------------------------------- Parte B: datos reales
function part_B(mu_grid; n_epochs)
    println("\n"*"="^72); println("PARTE B — DATOS REALES, QoQ (perfil del lag de maduración, ventana 1990–)"); println("="^72)
    P, I = load_window_qoq(from_year=1990); n = length(P)
    t = collect(0.0:0.25:0.25*(n-1)); PI = permutedims(hcat(P, I))
    oscale = maximum(abs.(PI))
    p_init = [0.5, 0.3, 0.5, 0.3, 0.6]
    @printf("ventana: %d trimestres (QoQ) │ grilla μ = %s\n", n, string(mu_grid))
    mus, losses, fits = profile_mu(t, PI, p_init, oscale, mu_grid; n_epochs=n_epochs)
    μ̂, Lmin, lo, hi = profile_summary(mus, losses)
    fbest = fits[argmin(losses)]
    @printf("\nPERFIL (real): mínimo en μ̂=%.2f trim (%.2f años) │ IC≈[%.2f, %.2f] │ k̂=%.3f │ L_min=%.4e\n",
            μ̂, μ̂/4, lo, hi, fbest.ode[5], Lmin)
    return mus, losses, μ̂, lo, hi, fbest.ode
end

function main(; smoke=false)
    mu_grid  = smoke ? MU_GRID_SMOKE : MU_GRID_FULL
    n_epochs = smoke ? 300 : 8000
    smoke && println(">>> SMOKE TEST: grilla=$(mu_grid), épocas=$(n_epochs) (NO es el run completo)\n")

    μt, musA, lossA, μ̂A, loA, hiA = part_A(mu_grid; n_epochs=n_epochs)
    musB, lossB, μ̂B, loB, hiB, odeB = part_B(mu_grid; n_epochs=n_epochs)

    # curvas de perfil
    open(joinpath(RESDIR, "p12_pinn_profile.csv"), "w") do io
        println(io, "caso,mu_trim,loss")
        for (m, l) in zip(musA, lossA); @printf(io, "sintetico,%.3f,%.6e\n", m, l); end
        for (m, l) in zip(musB, lossB); @printf(io, "real,%.3f,%.6e\n", m, l); end
    end
    # resumen
    open(joinpath(RESDIR, "p12_pinn_profile_summary.csv"), "w") do io
        println(io, "caso,lag_trim,lag_anios,ic_lo_trim,ic_hi_trim,k,note")
        @printf(io, "sintetico_verdadero,%.3f,%.3f,,,,lag conocido\n", μt, μt/4)
        @printf(io, "sintetico_perfil,%.3f,%.3f,%.3f,%.3f,,minimo del perfil\n", μ̂A, μ̂A/4, loA, hiA)
        @printf(io, "real_perfil,%.3f,%.3f,%.3f,%.3f,%.4f,QoQ ventana 1990-\n", μ̂B, μ̂B/4, loB, hiB, odeB[5])
    end

    println("\n"*"="^72); println("RESUMEN  (perfil → results/p12_pinn_profile.csv ; resumen → ..._summary.csv)")
    @printf("  Sintético: lag verdadero %.1f trim → perfil minimiza en %.2f trim (IC≈[%.2f,%.2f])\n",
            μt, μ̂A, loA, hiA)
    @printf("  Real:      lag de maduración %.2f trim (%.2f años), IC≈[%.2f,%.2f]\n",
            μ̂B, μ̂B/4, loB, hiB)
    println("="^72)
end

# --smoke en ARGS → grilla y épocas reducidas para verificar que corre end-to-end.
main(smoke = ("--smoke" in ARGS))
