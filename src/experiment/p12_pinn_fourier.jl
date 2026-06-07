# ============================================================
# p12_pinn_fourier.jl — PINN INVERSO con FEATURES DE FOURIER en la entrada
# ============================================================
# Variante de p11_pinn_maturation.jl. Misma física (cadena de maduración / kernel Gamma,
# m = convolución de I, se aprenden μ y k), mismo problema inverso bien puesto. El cambio
# es la ARQUITECTURA de la red sustituta:
#
#   p11:  t → (t−t0)/(T−t0)            (escalado lineal a [0,1])  → Dense·tanh → [P,I]
#   p12:  t → [sin(2π f_i t), cos(2π f_i t)]_{i=1..F}  (features de Fourier) → Dense·tanh → [P,I]
#
# Motivación: una MLP con entrada escalada sufre SPECTRAL BIAS — aprende primero las
# componentes de baja frecuencia y le cuesta representar oscilaciones rápidas en ventanas
# largas (la serie de ~145 trim). Cuando la red no representa la serie, el residuo físico
# no ata la dinámica y el lag de maduración colapsa a ~0. Mapear t a un banco de senos y
# cosenos de frecuencias log-espaciadas (random/Fourier features) entrega a las capas densas
# una base ya oscilatoria, mitigando el sesgo espectral y permitiendo ajustar la ventana
# completa. Las frecuencias se eligen log-espaciadas para cubrir desde el modo lento
# (~1 ciclo en la ventana) hasta el Nyquist de la grilla trimestral.
#
# Modelo físico (idéntico a p11):
#   m(t) = Σⱼ wⱼ(μ) · I(t − j)          (inversión madurada; wⱼ = kernel Gamma, media μ)
#   dP/dt = c·I − d·P − k·m              (ganancia: + demanda actual, − sobreacumulación madurada)
#   dI/dt = a·P − b·I                    (acelerador)
#   p̂ = [a,b,c,d,k,logμ];  LAG DE MADURACIÓN = μ (trimestres);  μ = exp(logμ) > 0
#
# PINN inverso:  L = L_data + λ·L_phys, λ=1; derivada de la red por diferencias centradas;
# se optimizan JUNTOS los pesos de la red y p̂; Adam; seed fijo; init lejos del verdadero.
#
# Transform: QoQ (pct_change(1)) sobre los niveles PROFITS/INVESTMENT — el insesgado
# (YoY = pct_change(4) infla ~2× el feedback en lag−4; ver notes/STATUS.md §4).
#
# Parte A: VALIDACIÓN EN SINTÉTICO — recupera un lag conocido midiendo error %.
# Parte B: DATOS REALES — estima el lag de maduración sobre la ventana 1990–.
#
# Corre:  julia --project=julia src/experiment/p12_pinn_fourier.jl
# ============================================================
using DifferentialEquations, Lux, Optimisers, Zygote, Random, Statistics, Printf, DelimitedFiles

const SHAPE  = 4.0          # forma del kernel Gamma (fija; el ancho no es identificable, sí la media μ)
const KMAX   = 10           # rezagos del kernel (trimestres)
const λ_phys = 1.0
const N_FOURIER = 8         # nº de frecuencias de Fourier (cada una aporta un sin y un cos)
const REPO   = abspath(joinpath(@__DIR__, "..", ".."))
const CSVP   = joinpath(REPO, "data", "processed", "lotka_volterra.csv")
const RESDIR = joinpath(REPO, "results"); mkpath(RESDIR)

# ---------------------------------------------------------------- kernel Gamma (diferenciable en μ)
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

# ---------------------------------------------------------------- features de Fourier
# t normalizado a τ∈[0,1]; frecuencias log-espaciadas entre f_min y f_max ciclos en la
# ventana. f_min≈1 (modo más lento), f_max≈n/2 (Nyquist de la grilla trimestral). Devuelve
# [sin(2π f_i τ), cos(2π f_i τ)]  → 2·N_FOURIER features (determinista por f log-espaciadas).
# frecuencias log-espaciadas (constantes: dependen sólo de nfreq y fmax, no de θ)
make_freqs(nfreq, fmax) = collect(exp.(range(log(1.0), log(fmax); length=nfreq)))

# t (escalar en vector) → [sin(2π f τ), cos(2π f τ)]; freqs constante capturado por la red.
# τ = (t−t0)/(T−t0). Sin mutación y sin construir el rango adentro (Zygote-friendly).
function fourier_features(t, t0, T, freqs)
    τ = (t[1] - t0) / (T - t0)
    return vcat([[sin(2π * f * τ), cos(2π * f * τ)] for f in freqs]...)
end

# ---------------------------------------------------------------- red con entrada de Fourier
function build_nn(t0, T, oscale; nfreq=N_FOURIER, fmax=20.0)
    freqs = make_freqs(nfreq, fmax)
    Chain(WrappedFunction(t -> fourier_features(t, t0, T, freqs)),
          Dense(2*nfreq=>32, tanh), Dense(32=>32, tanh), Dense(32=>16, tanh), Dense(16=>2),
          WrappedFunction(x -> oscale .* x))
end
nn_dudt(nn, t, ps, st; h=1e-4) = (nn([t+h], ps, st)[1] .- nn([t-h], ps, st)[1]) ./ (2h)

# ---------------------------------------------------------------- entrenamiento del PINN inverso
function train(t, PI_obs, p_init, oscale; n_epochs=8000, seed=42, lr=1e-2, fmax=20.0, verbose=true)
    n = length(t); Δ = t[2]-t[1]
    nn = build_nn(t[1], t[end], oscale; fmax=fmax)
    nn_ps, st = Lux.setup(MersenneTwister(seed), nn); st = Lux.testmode(st)
    nn_ps = Lux.f64(nn_ps)
    theta = (nn = nn_ps, ode = copy(p_init))

    function comps(th)
        ps = th.nn
        a,b,c,d,k = th.ode[1],th.ode[2],th.ode[3],th.ode[4],th.ode[5]
        μ = exp(th.ode[6]); w = gamma_kernel(μ)
        P = [nn([t[i]], ps, st)[1][1] for i in 1:n]
        I = [nn([t[i]], ps, st)[1][2] for i in 1:n]
        data = mean((P .- PI_obs[1,:]).^2 .+ (I .- PI_obs[2,:]).^2)
        # m = kernel * I (inversión madurada)
        m = [sum(w[j+1]*I[i-j] for j in 0:min(KMAX, i-1)) for i in 1:n]
        # residuo en puntos interiores (necesitan derivada central + historia del kernel)
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
        if verbose && epoch % 100 == 0
            dl, pl = comps(theta)
            @printf("  época %5d │ L_data=%.3e │ L_phys=%.3e │ lag μ=%.2f trim │ k=%.3f\n",
                    epoch, dl, pl, exp(theta.ode[6]), theta.ode[5])
        end
    end
    return theta, nn, st
end

# ---------------------------------------------------------------- Parte A: sintético
function part_A(; n_epochs=8000)
    println("="^72); println("PARTE A — VALIDACIÓN EN SINTÉTICO (recuperar lag conocido)"); println("="^72)
    θ_true = 0.25                                   # años/etapa → lag = 4·0.25·4(trim/año)=4 trim = 1 año
    μ_true_q = 4.0                                  # lag verdadero en trimestres (1 año)
    p_chain = [0.9, 0.4, 0.6, 0.4, 1.1, θ_true]
    u0 = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]; tspan = (0.0, 24.0)
    sol = solve(ODEProblem(chain_rhs!, u0, tspan, p_chain), Tsit5(), saveat=0.25, abstol=1e-8, reltol=1e-8)
    t = collect(0.0:0.25:24.0); n = length(t)
    PI = hcat([sol(τ)[1:2] for τ in t]...)
    PI .+= 0.03*std(PI).*randn(MersenneTwister(0), size(PI))
    oscale = maximum(abs.(PI))
    # fmax ~ Nyquist de la grilla: hasta ~n/2 ciclos en la ventana
    fmax = n / 2.0
    p_init = [0.5, 0.2, 0.3, 0.2, 0.5, log(8.0)]    # lag init = 8 trim (lejos del 4 real)
    @printf("verdadero: lag μ=%.1f trim (1 año), k=%.2f │ init: lag=%.1f trim, k=%.2f │ Fourier: %d freqs, fmax=%.0f\n",
            μ_true_q, p_chain[5], exp(p_init[6]), p_init[5], N_FOURIER, fmax)
    theta, _, _ = train(t, PI, p_init, oscale; n_epochs=n_epochs, fmax=fmax)
    μ̂ = exp(theta.ode[6])
    @printf("\nRECUPERACIÓN: lag verdadero=%.2f trim │ estimado=%.2f trim │ error=%.1f%%  │ k̂=%.3f\n",
            μ_true_q, μ̂, 100*abs(μ̂-μ_true_q)/μ_true_q, theta.ode[5])
    return μ_true_q, μ̂, theta.ode
end

# ---------------------------------------------------------------- Parte B: datos reales (QoQ)
# QoQ = pct_change(1) sobre los niveles PROFITS / INVESTMENT (transform insesgado; ver §4).
function load_window(; from_year=1990)
    raw, h = readdlm(CSVP, ',', header=true); hv = vec(h)
    di=findfirst(==("DATE"),hv); pi_=findfirst(==("PROFITS"),hv); ii=findfirst(==("INVESTMENT"),hv)
    rows=[r for r in 1:size(raw,1) if raw[r,pi_]!="" && raw[r,ii]!=""]
    dts=[string(raw[r,di]) for r in rows]
    Plev=Float64[raw[r,pi_] for r in rows]; Ilev=Float64[raw[r,ii] for r in rows]
    # QoQ: variación trimestral porcentual (pct_change(1)); descarta el primer punto (NaN)
    Pqoq = 100 .* (Plev[2:end] .- Plev[1:end-1]) ./ Plev[1:end-1]
    Iqoq = 100 .* (Ilev[2:end] .- Ilev[1:end-1]) ./ Ilev[1:end-1]
    dq   = dts[2:end]
    keep=[parse(Int,d[1:4])>=from_year for d in dq]
    z(v)=(v.-mean(v))./std(v)
    return z(Pqoq[keep]), z(Iqoq[keep])
end

function part_B(; from_year=1990, n_epochs=8000)
    println("\n"*"="^72); println("PARTE B — DATOS REALES QoQ (lag de maduración, ventana 1990–)"); println("="^72)
    P,I = load_window(from_year=from_year); n=length(P)
    t = collect(0.0:0.25:0.25*(n-1)); PI = permutedims(hcat(P,I))
    oscale = maximum(abs.(PI))
    fmax = n / 2.0
    p_init = [0.5, 0.3, 0.5, 0.3, 0.6, log(6.0)]
    @printf("ventana: %d trimestres │ init lag=%.1f trim │ Fourier: %d freqs, fmax=%.0f\n",
            n, exp(p_init[6]), N_FOURIER, fmax)
    theta, nn, st = train(t, PI, p_init, oscale; n_epochs=n_epochs, fmax=fmax)
    μ̂ = exp(theta.ode[6])
    Pp=[nn([τ],theta.nn,st)[1][1] for τ in t]; Ip=[nn([τ],theta.nn,st)[1][2] for τ in t]
    r2 = 1 - (sum((P.-Pp).^2)+sum((I.-Ip).^2))/(sum((P.-mean(P)).^2)+sum((I.-mean(I)).^2))
    @printf("\nESTIMACIÓN (real, QoQ): lag de maduración μ = %.2f trim (%.2f años) │ k̂=%.3f │ R²(P,I)=%.2f\n",
            μ̂, μ̂/4, theta.ode[5], r2)
    return μ̂, theta.ode, r2
end

function main(; n_epochs=8000)
    μt, μh, _   = part_A(; n_epochs=n_epochs)
    μr, pr, r2  = part_B(; n_epochs=n_epochs)
    open(joinpath(RESDIR, "p12_pinn_fourier_estimates.csv"), "w") do io
        println(io, "caso,lag_trim,lag_anios,a,b,c,d,k,r2")
        @printf(io, "sintetico_verdadero,%.3f,%.3f,,,,,,\n", μt, μt/4)
        @printf(io, "sintetico_recuperado,%.3f,%.3f,,,,,,\n", μh, μh/4)
        @printf(io, "real,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
                μr, μr/4, pr[1],pr[2],pr[3],pr[4],pr[5], r2)
    end
    println("\n"*"="^72); println("RESUMEN  (guardado en results/p12_pinn_fourier_estimates.csv)")
    @printf("  Sintético: lag verdadero %.1f trim → recuperado %.2f trim (error %.1f%%)\n",
            μt, μh, 100*abs(μh-μt)/μt)
    @printf("  Real (QoQ): lag de maduración %.2f trim (%.2f años), R²=%.2f\n", μr, μr/4, r2)
    println("="^72)
end

# Run completo: main()  (8000 épocas). Para smoke test pasar n_epochs chico, p.ej. main(n_epochs=500).
main()
