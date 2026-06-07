# ============================================================
# p12_pinn_percrisis.jl — PINN INVERSO por VENTANAS CORTAS por crisis con lag POOLEADO
# ============================================================
# Variante de p11_pinn_maturation.jl. El ajuste sobre la ventana larga (≈145 trim continuos)
# sufre spectral bias: la red de baja frecuencia no representa la serie completa y el lag de
# maduración μ colapsa a ~0 (queda sin anclar). Hipótesis del fix probada acá:
#
#   ajustar CADA vecindad de crisis por separado (ventanas cortas, ±NB trim alrededor del
#   valle de la recesión) con su PROPIA red, y POOLEAR el lag μ COMPARTIÉNDOLO entre crisis.
#
# Las ventanas cortas tienen contenido de frecuencia que la red sí puede representar (evita
# el spectral bias); el pooling de μ usa las ~9 respuestas de crisis en conjunto para anclar
# un único lag de maduración, en vez de pedirle a una sola ventana larga que lo identifique.
#
# Modelo físico (idéntico a p11; cadena de maduración resuelta por convolución del kernel Gamma):
#   m(t) = Σⱼ wⱼ(μ) · I(t − j)          (inversión madurada; wⱼ = kernel Gamma, media μ)
#   dP/dt = c·I − d·P − k·m              (ganancia: + demanda actual, − sobreacumulación madurada)
#   dI/dt = a·P − b·I                    (acelerador)
#
# Parámetros COMPARTIDOS entre todas las crisis: p̂ = [a,b,c,d,k,logμ]; lag = μ = exp(logμ).
# Parámetros POR CRISIS: los pesos de una red propia  t ↦ [P(t), I(t)]  por ventana.
# El pooling de los parámetros físicos (en particular μ y k) es lo que da poder estadístico:
# cada ventana contribuye su residuo de la ODE al mismo μ, k.
#
# Datos: QoQ = pct_change(1) sobre los NIVELES PROFITS / INVESTMENT del CSV (no YoY); cada
# ventana se z-score-a por separado (la red de cada ventana ve su propia escala).
#
# Parte A: VALIDACIÓN EN SINTÉTICO — varias ventanas cortas con un lag común conocido;
#          mide el error % de recuperación del lag pooleado.
# Parte B: DATOS REALES — estima el lag de maduración pooleando las vecindades de crisis.
#
# Corre:  julia --project=julia src/experiment/p12_pinn_percrisis.jl
# ============================================================
using DifferentialEquations, Lux, Optimisers, Zygote, Random, Statistics, Printf, DelimitedFiles, Dates

const SHAPE   = 4.0         # forma del kernel Gamma (fija; el ancho no es identificable, sí la media μ)
const KMAX    = 10          # rezagos del kernel (trimestres)
const λ_phys  = 1.0
const NB      = 8           # semiancho de la ventana por crisis (±NB trimestres alrededor del valle)
const REPO    = abspath(joinpath(@__DIR__, "..", ".."))
const CSVP    = joinpath(REPO, "data", "processed", "lotka_volterra.csv")
const RESDIR  = joinpath(REPO, "results"); mkpath(RESDIR)

# Valles (troughs) de las recesiones NBER, fin de trimestre (de common.py). Se excluyen
# 2008 y 2020 (fuera del régimen de amplitud; ver EXPERIMENT.md).
const TROUGHS = ["1949-12-31","1954-06-30","1958-06-30","1961-03-31","1970-12-31",
                 "1975-03-31","1980-09-30","1982-12-31","1991-03-31","2001-12-31"]

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

# ---------------------------------------------------------------- red (una por ventana)
function build_nn(t0, T, oscale)
    Chain(WrappedFunction(t -> (t .- t0) ./ (T - t0)),
          Dense(1=>32, tanh), Dense(32=>32, tanh), Dense(32=>16, tanh), Dense(16=>2),
          WrappedFunction(x -> oscale .* x))
end
nn_dudt(nn, t, ps, st; h=1e-4) = (nn([t+h], ps, st)[1] .- nn([t-h], ps, st)[1]) ./ (2h)

# Una ventana: tiempos, observaciones z-score (2×n), su red, estado, escala de salida.
struct Window
    t::Vector{Float64}
    PI::Matrix{Float64}
    nn::Any
    st::Any
    oscale::Float64
end

function make_window(t, PI; seed)
    oscale = maximum(abs.(PI))
    nn = build_nn(t[1], t[end], oscale)
    ps, st = Lux.setup(MersenneTwister(seed), nn); st = Lux.testmode(st)
    ps = Lux.f64(ps)
    return Window(t, PI, nn, st, oscale), ps
end

# ---------------------------------------------------------------- residuo (data + física) de UNA ventana
# Comparte los parámetros físicos ODE (incl. μ vía logμ); usa la red propia de la ventana.
function window_comps(win::Window, ps_nn, ode)
    nn, st, t, PI = win.nn, win.st, win.t, win.PI
    n = length(t)
    a,b,c,d,k = ode[1],ode[2],ode[3],ode[4],ode[5]
    μ = exp(ode[6]); w = gamma_kernel(μ)
    P = [nn([t[i]], ps_nn, st)[1][1] for i in 1:n]
    I = [nn([t[i]], ps_nn, st)[1][2] for i in 1:n]
    data = mean((P .- PI[1,:]).^2 .+ (I .- PI[2,:]).^2)
    m = [sum(w[j+1]*I[i-j] for j in 0:min(KMAX, i-1)) for i in 1:n]
    phys = mean(map((KMAX+1):(n-1)) do i
        du = nn_dudt(nn, t[i], ps_nn, st)
        rP = du[1] - (c*I[i] - d*P[i] - k*m[i])
        rI = du[2] - (a*P[i] - b*I[i])
        rP^2 + rI^2
    end)
    return data, λ_phys*phys
end

# ---------------------------------------------------------------- entrenamiento POOLEADO
# theta = (nns = (ps_1, ps_2, ...), ode = [a,b,c,d,k,logμ] compartido entre todas las ventanas).
function train_pooled(wins::Vector{Window}, ps_list, p_init; n_epochs=8000, lr=1e-2, verbose=true)
    theta = (nns = Tuple(ps_list), ode = copy(p_init))
    function loss(th)
        tot = 0.0
        for (j, win) in enumerate(wins)
            dl, pl = window_comps(win, th.nns[j], th.ode)
            tot += dl + pl
        end
        return tot / length(wins)
    end
    opt = Optimisers.setup(Adam(lr), theta)
    for epoch in 1:n_epochs
        _, g = Zygote.withgradient(loss, theta)
        opt, theta = Optimisers.update(opt, theta, g[1])
        if verbose && (epoch % max(1, n_epochs ÷ 8) == 0 || epoch == 1)
            dl = mean(window_comps(w, theta.nns[j], theta.ode)[1] for (j,w) in enumerate(wins))
            pl = mean(window_comps(w, theta.nns[j], theta.ode)[2] for (j,w) in enumerate(wins))
            @printf("  época %5d │ L_data=%.3e │ L_phys=%.3e │ lag μ=%.2f trim │ k=%.3f\n",
                    epoch, dl, pl, exp(theta.ode[6]), theta.ode[5])
        end
    end
    return theta
end

# ---------------------------------------------------------------- Parte A: sintético (varias ventanas, lag común)
function synth_window(seed_data; θ_true, p_chain, half=NB, Δ=0.25)
    u0 = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]; tspan = (0.0, 40.0)
    sol = solve(ODEProblem(chain_rhs!, u0, tspan, [p_chain...; θ_true]), Tsit5(),
                saveat=Δ, abstol=1e-8, reltol=1e-8)
    # recorta una vecindad de 2*half+1 trim alrededor de un punto interior aleatorio
    rng = MersenneTwister(seed_data)
    grid = collect(0.0:Δ:40.0)
    c0 = rand(rng, (KMAX+half+2):(length(grid)-half-1))   # deja historia para el kernel
    idx = (c0-half):(c0+half)
    t = grid[idx]
    PI = hcat([sol(τ)[1:2] for τ in t]...)
    PI .+= 0.03*std(PI).*randn(rng, size(PI))
    # z-score por ventana (como en los datos reales)
    for r in 1:2
        PI[r,:] .= (PI[r,:] .- mean(PI[r,:])) ./ std(PI[r,:])
    end
    t = t .- t[1]
    return t, PI
end

function part_A(; n_epochs=8000, n_windows=6)
    println("="^72); println("PARTE A — SINTÉTICO: lag COMÚN pooleado en ventanas cortas"); println("="^72)
    θ_true = 0.25; μ_true_q = 4.0
    p_chain = [0.9, 0.4, 0.6, 0.4, 1.1]            # a,b,c,d,k (lag = 4·θ_true·… → μ_true_q=4 trim)
    wins = Window[]; ps_list = Any[]
    for s in 1:n_windows
        t, PI = synth_window(100+s; θ_true=θ_true, p_chain=p_chain)
        w, ps = make_window(t, PI; seed=42+s)
        push!(wins, w); push!(ps_list, ps)
    end
    p_init = [0.5, 0.2, 0.3, 0.2, 0.5, log(8.0)]   # lag init = 8 trim (lejos del 4 real)
    @printf("verdadero: lag μ=%.1f trim, k=%.2f │ init: lag=%.1f trim, k=%.2f │ %d ventanas de %d trim\n",
            μ_true_q, p_chain[5], exp(p_init[6]), p_init[5], n_windows, 2NB+1)
    theta = train_pooled(wins, ps_list, p_init; n_epochs=n_epochs)
    μ̂ = exp(theta.ode[6])
    @printf("\nRECUPERACIÓN: lag verdadero=%.2f trim │ estimado=%.2f trim │ error=%.1f%% │ k̂=%.3f\n",
            μ_true_q, μ̂, 100*abs(μ̂-μ_true_q)/μ_true_q, theta.ode[5])
    return μ_true_q, μ̂, theta.ode
end

# ---------------------------------------------------------------- Parte B: datos reales (vecindades de crisis)
function load_qoq()
    raw, h = readdlm(CSVP, ',', header=true); hv = vec(h)
    di=findfirst(==("DATE"),hv); pi=findfirst(==("PROFITS"),hv); ii=findfirst(==("INVESTMENT"),hv)
    dts = [Date(string(raw[r,di])) for r in 1:size(raw,1)]
    Plv = Float64[raw[r,pi] for r in 1:size(raw,1)]
    Ilv = Float64[raw[r,ii] for r in 1:size(raw,1)]
    qoq(v) = [NaN; (v[2:end] .- v[1:end-1]) ./ v[1:end-1]]    # pct_change(1) sobre niveles
    return dts, qoq(Plv), qoq(Ilv)
end

function crisis_windows(; half=NB)
    dts, Pq, Iq = load_qoq()
    wins = Window[]; ps_list = Any[]; used = String[]
    for (s, tr) in enumerate(TROUGHS)
        c = findfirst(==(Date(tr)), dts)
        c === nothing && continue
        lo, hi = c-half, c+half
        (lo < 2 || hi > length(dts)) && continue          # 1949 no entra (sin historia previa)
        idx = lo:hi
        P = Pq[idx]; I = Iq[idx]
        (any(isnan, P) || any(isnan, I)) && continue
        z(v) = (v .- mean(v)) ./ std(v)
        PI = permutedims(hcat(z(P), z(I)))
        t = collect(0.0:0.25:0.25*(length(idx)-1))
        w, ps = make_window(t, PI; seed=7+s)
        push!(wins, w); push!(ps_list, ps); push!(used, tr[1:4])
    end
    return wins, ps_list, used
end

function part_B(; n_epochs=8000)
    println("\n"*"="^72); println("PARTE B — DATOS REALES: lag pooleado sobre vecindades de crisis (QoQ)"); println("="^72)
    wins, ps_list, used = crisis_windows()
    @printf("crisis usadas (%d): %s │ ventana ±%d trim (%d obs c/u)\n",
            length(used), join(used, ", "), NB, 2NB+1)
    p_init = [0.5, 0.3, 0.5, 0.3, 0.6, log(6.0)]
    theta = train_pooled(wins, ps_list, p_init; n_epochs=n_epochs)
    μ̂ = exp(theta.ode[6])
    # R² promedio sobre las ventanas
    r2s = Float64[]
    for (j, w) in enumerate(wins)
        n = length(w.t)
        Pp = [w.nn([τ], theta.nns[j], w.st)[1][1] for τ in w.t]
        Ip = [w.nn([τ], theta.nns[j], w.st)[1][2] for τ in w.t]
        P = w.PI[1,:]; I = w.PI[2,:]
        ss = sum((P.-Pp).^2)+sum((I.-Ip).^2); tt = sum((P.-mean(P)).^2)+sum((I.-mean(I)).^2)
        push!(r2s, 1 - ss/tt)
    end
    r2 = mean(r2s)
    @printf("\nESTIMACIÓN (real): lag de maduración POOLEADO μ = %.2f trim (%.2f años) │ k̂=%.3f │ R²̄(P,I)=%.2f\n",
            μ̂, μ̂/4, theta.ode[5], r2)
    return μ̂, theta.ode, r2, length(used)
end

function main(; n_epochs=8000)
    μt, μh, _      = part_A(; n_epochs=n_epochs)
    μr, pr, r2, nc = part_B(; n_epochs=n_epochs)
    open(joinpath(RESDIR, "p12_pinn_percrisis_estimates.csv"), "w") do io
        println(io, "caso,n_crisis,lag_trim,lag_anios,a,b,c,d,k,r2")
        @printf(io, "sintetico_verdadero,,%.3f,%.3f,,,,,,\n", μt, μt/4)
        @printf(io, "sintetico_recuperado,,%.3f,%.3f,,,,,,\n", μh, μh/4)
        @printf(io, "real,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
                nc, μr, μr/4, pr[1],pr[2],pr[3],pr[4],pr[5], r2)
    end
    println("\n"*"="^72); println("RESUMEN  (guardado en results/p12_pinn_percrisis_estimates.csv)")
    @printf("  Sintético: lag verdadero %.1f trim → recuperado %.2f trim (error %.1f%%)\n",
            μt, μh, 100*abs(μh-μt)/μt)
    @printf("  Real:      lag de maduración pooleado %.2f trim (%.2f años) sobre %d crisis, R²̄=%.2f\n",
            μr, μr/4, nc, r2)
    println("="^72)
end

# Run completo: main(n_epochs=8000). Para SMOKE TEST se invoca con pocas épocas vía ENV.
const SMOKE = get(ENV, "P12_SMOKE", "0") == "1"
main(; n_epochs = SMOKE ? 500 : 8000)
