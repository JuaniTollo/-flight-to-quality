# ============================================================
# p11_pinn_maturation.jl — PINN INVERSO del modelo de cadena de maduración
# ============================================================
# Modelo físico del ciclo de sobreacumulación profit→investment con delay de
# maduración DISTRIBUIDO (cadena de n=4 etapas; linear chain trick ⇒ kernel Gamma):
#
#   dP/dt = c·I − d·P − k·m₄        (ganancia: + demanda actual, − sobreacumulación madurada)
#   dI/dt = a·P − b·I               (acelerador)
#   dm₁/dt = (I  − m₁)/θ            ┐ cadena de maduración: m₄ = inversión retardada
#   dm₂/dt = (m₁ − m₂)/θ            │ por un kernel Gamma de media 4·θ y dispersión √4·θ
#   dm₃/dt = (m₂ − m₃)/θ            │ ⇒ el LAG DE MADURACIÓN estimado = 4·θ
#   dm₄/dt = (m₃ − m₄)/θ            ┘
#
# Estado u = [P, I, m₁, m₂, m₃, m₄]. P,I son OBSERVADOS; m_j son LATENTES (sin datos).
#
# PINN inverso (mismas consideraciones que el experimento canónico de PINN inverso):
#   - red  t ↦ u_θ(t) ∈ ℝ⁶  con escalado de entrada (t→[0,1]) y de salida; tanh; 32-32-16
#   - pérdida  L = L_data + λ·L_phys,  λ=1
#       L_data = mean‖u_θ[1:2](tⱼ) − [P,I]_obs(tⱼ)‖²   (SOLO sobre P,I observados)
#       L_phys = mean‖du_θ/dt(tᵢ) − f(u_θ(tᵢ), p̂)‖²    (residuo de las 6 ecuaciones)
#   - se optimizan JUNTOS los pesos de la red y los parámetros ODE p̂=[a,b,c,d,k,logθ]
#   - θ = exp(logθ) > 0 ; derivada de la red por diferencias finitas centradas
#   - puntos de colocación log+uniforme ; Adam ; seed fijo ; params init lejos
#
# Parte A: VALIDACIÓN EN SINTÉTICO — recupera un lag conocido (4·θ*) midiendo error %.
# Parte B: DATOS REALES — estima el lag de maduración sobre una ventana representativa.
#
# Corre:  julia --project=julia \
#               src/experiment/p11_pinn_maturation.jl
# ============================================================
using DifferentialEquations, Lux, Optimisers, Zygote, Random, Statistics, Printf, DelimitedFiles

const N_STAGES = 4          # etapas de maduración (lag medio = N_STAGES·θ)
const λ_phys   = 1.0
const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const CSVP     = joinpath(REPO, "data", "processed", "lotka_volterra.csv")

# ---------------------------------------------------------------- modelo (RHS)
function maturation_rhs(u, p)
    P, I, m1, m2, m3, m4 = u
    a, b, c, d, k = p[1], p[2], p[3], p[4], p[5]
    θ = exp(p[6])
    return [c*I - d*P - k*m4,
            a*P - b*I,
            (I  - m1)/θ,
            (m1 - m2)/θ,
            (m2 - m3)/θ,
            (m3 - m4)/θ]
end

function maturation_ode!(du, u, p, t)
    du .= maturation_rhs(u, p)
end

# ---------------------------------------------------------------- red + PINN inverso
function build_nn(t0, T, out_scale)
    scale_inp = WrappedFunction(t -> (t .- t0) ./ (T - t0))
    scale_out = WrappedFunction(x -> out_scale .* x)
    Chain(scale_inp, Dense(1=>32, tanh), Dense(32=>32, tanh),
          Dense(32=>16, tanh), Dense(16=>6), scale_out)
end

nn_dudt(nn, t, ps, st; h=1e-4) =
    (nn([t+h], ps, st)[1] .- nn([t-h], ps, st)[1]) ./ (2h)

function train_inverse_pinn(t_obs, PI_obs, t_col, p_init, out_scale;
                            n_epochs=10000, seed=42, lr=1e-3, verbose=true)
    t0, T = first(t_col), last(t_col)
    nn = build_nn(t0, T, out_scale)
    rng = MersenneTwister(seed)
    nn_ps, st = Lux.setup(rng, nn); st = Lux.testmode(st)
    theta = (nn = nn_ps, ode = copy(p_init))
    M = length(t_obs)

    function loss_components(th)
        ps = th.nn
        data = mean(map(1:M) do j
            u = nn([t_obs[j]], ps, st)[1]
            (u[1] - PI_obs[1, j])^2 + (u[2] - PI_obs[2, j])^2
        end)
        phys = λ_phys * mean(map(t_col) do ti
            u = nn([ti], ps, st)[1]
            r = nn_dudt(nn, ti, ps, st) .- maturation_rhs(u, th.ode)
            sum(r .^ 2)
        end)
        return data, phys
    end
    loss(th) = sum(loss_components(th))

    opt = Optimisers.setup(Adam(lr), theta)
    for epoch in 1:n_epochs
        lval, grads = Zygote.withgradient(loss, theta)
        opt, theta = Optimisers.update(opt, theta, grads[1])
        if verbose && epoch % 2000 == 0
            d, p = loss_components(theta)
            lag = N_STAGES * exp(theta.ode[6])
            @printf("  época %6d │ L_data=%.3e │ L_phys=%.3e │ lag=4θ=%.2f a │ k=%.3f\n",
                    epoch, d, p, lag, theta.ode[5])
        end
    end
    return theta, nn, st
end

# ---------------------------------------------------------------- Parte A: sintético
function part_A()
    println("="^72)
    println("PARTE A — VALIDACIÓN EN SINTÉTICO (recuperar un lag de maduración conocido)")
    println("="^72)
    # parámetros verdaderos (desconocidos para el PINN); θ*=0.25 ⇒ lag*=4·0.25=1.0 año
    p_true = [0.9, 0.4, 0.6, 0.4, 1.1, log(0.25)]
    lag_true = N_STAGES * exp(p_true[6])
    u0 = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    tspan = (0.0, 15.0)
    prob = ODEProblem(maturation_ode!, u0, tspan, p_true)
    sol  = solve(prob, Tsit5(), saveat=0.25, abstol=1e-8, reltol=1e-8)

    M = 40
    t_obs = collect(range(tspan[1], tspan[2], length=M))
    rng = MersenneTwister(0)
    PI_obs = hcat([sol(t)[1:2] for t in t_obs]...)
    PI_obs .+= 0.03 .* std(PI_obs) .* randn(rng, size(PI_obs))   # ruido leve
    out_scale = maximum(abs.(PI_obs))

    Nc = 100
    t_col = sort([10.0 .^ range(log10(1e-3), log10(tspan[2]), length=div(Nc,2));
                  collect(range(tspan[1], tspan[2], length=div(Nc,2)))])

    # init LEJOS del verdadero (incluido θ): lag_init = 4·exp(log 0.6)=2.4a
    p_init = [0.5, 0.2, 0.3, 0.2, 0.5, log(0.6)]
    @printf("verdadero: lag=4θ=%.2f a, k=%.2f │ init: lag=%.2f a, k=%.2f\n",
            lag_true, p_true[5], N_STAGES*exp(p_init[6]), p_init[5])

    theta, nn, st = train_inverse_pinn(t_obs, PI_obs, t_col, p_init, out_scale; n_epochs=12000)
    lag_hat = N_STAGES * exp(theta.ode[6])
    @printf("\nRECUPERACIÓN: lag verdadero=%.3f a │ estimado=%.3f a │ error=%.1f%%\n",
            lag_true, lag_hat, 100*abs(lag_hat-lag_true)/lag_true)
    @printf("              k verdadero=%.3f │ estimado=%.3f\n", p_true[5], theta.ode[5])
    return lag_true, lag_hat
end

# ---------------------------------------------------------------- Parte B: datos reales
function load_window(; from_year=1990)
    raw, h = readdlm(CSVP, ',', header=true); hv = vec(h)
    di = findfirst(==("DATE"), hv)
    xi = findfirst(==("PROFITS_YOY"), hv); yi = findfirst(==("INVEST_YOY"), hv)
    rows = [r for r in 1:size(raw,1) if !isempty(string(raw[r,di])) &&
            raw[r,xi] != "" && raw[r,yi] != ""]
    dates = [string(raw[r,di]) for r in rows]
    P = Float64[raw[r,xi] for r in rows]; I = Float64[raw[r,yi] for r in rows]
    keep = [parse(Int, d[1:4]) >= from_year for d in dates]
    P, I = P[keep], I[keep]
    zscore(v) = (v .- mean(v)) ./ std(v)
    return zscore(P), zscore(I)
end

function part_B()
    println("\n" * "="^72)
    println("PARTE B — DATOS REALES (estimar el lag de maduración, ventana 1990–)")
    println("="^72)
    P, I = load_window(from_year=1990)
    n = length(P)
    t_obs = collect(0.0:0.25:0.25*(n-1))           # años, trimestral
    PI_obs = permutedims(hcat(P, I))               # (2, n)
    out_scale = maximum(abs.(PI_obs))
    T = last(t_obs)
    Nc = 120
    t_col = sort([10.0 .^ range(log10(1e-3), log10(T), length=div(Nc,2));
                  collect(range(0.0, T, length=div(Nc,2)))])
    p_init = [0.5, 0.3, 0.5, 0.3, 0.6, log(0.5)]   # lag_init=2a
    @printf("ventana: %d trimestres (%.1f años) │ init lag=%.2f a\n", n, T, N_STAGES*exp(p_init[6]))
    theta, nn, st = train_inverse_pinn(t_obs, PI_obs, t_col, p_init, out_scale; n_epochs=12000)
    lag_hat = N_STAGES * exp(theta.ode[6])
    # R² de datos
    Pp = [nn([t], theta.nn, st)[1][1] for t in t_obs]
    Ip = [nn([t], theta.nn, st)[1][2] for t in t_obs]
    r2 = 1 - (sum((P.-Pp).^2)+sum((I.-Ip).^2)) / (sum((P.-mean(P)).^2)+sum((I.-mean(I)).^2))
    @printf("\nESTIMACIÓN (datos reales): lag de maduración = 4θ = %.2f años (%.1f trimestres)\n",
            lag_hat, lag_hat*4)
    @printf("                           k (feedback) = %.3f │ R²(P,I) = %.2f\n", theta.ode[5], r2)
    return lag_hat, r2
end

function main()
    lt, lh = part_A()
    lbh, r2 = part_B()
    println("\n" * "="^72)
    println("RESUMEN")
    @printf("  Sintético: lag verdadero %.2f a → recuperado %.2f a (error %.1f%%)\n",
            lt, lh, 100*abs(lh-lt)/lt)
    @printf("  Real:      lag de maduración estimado %.2f años (~%.0f trimestres), R²=%.2f\n",
            lbh, lbh*4, r2)
    println("="^72)
end

main()
