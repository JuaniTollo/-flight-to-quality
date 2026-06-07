# ============================================================
# oscillation.jl — Pieza 1: la oscilación característica (Julia)
# ============================================================
# Ajusta FN y LV por solver single-shooting sobre una VENTANA REPRESENTATIVA (un par de
# ciclos limpios, pre-2008) — el ajuste full-series degeneraba a casi-punto-fijo. Exporta
# la trayectoria ajustada y los datos para que Python grafique (trayectoria + fase).
#
# Corre:  julia --project=<entorno Julia con DifferentialEquations, Lux, Zygote, Optim> \
#               src/experiment/oscillation.jl
# ============================================================
include(joinpath(@__DIR__, "models.jl"))   # FN, LV, osolve, fit_solver, load_yoy, CSVP, OUTDIR

const WIN_LO = Date("1991-01-01")           # expansión de los '90 + recesión 2001: ciclos limpios
const WIN_HI = Date("2002-12-31")

function main_osc()
    dates, prof, inv = load_yoy(CSVP)
    mask = (dates .>= WIN_LO) .& (dates .<= WIN_HI)
    d, xt, yt = dates[mask], prof[mask], inv[mask]
    decyr(dd) = year(dd) + (dayofyear(dd) - 1) / 365.25
    t = [Dates.value(dd - d[1]) / 365.25 for dd in d]
    μx, σx = mean(xt), std(xt); μy, σy = mean(yt), std(yt)

    open(joinpath(OUTDIR, "julia_oscillation.csv"), "w") do io
        println(io, "kind,decyear,profits,investment")
        for (dd, x, y) in zip(d, xt, yt)
            @printf(io, "data,%.4f,%.4f,%.4f\n", decyr(dd), x, y)
        end
        for m in (FN, LV)
            offset = 0.0
            if m.needs_offset
                zx = (xt .- μx) ./ σx; zy = (yt .- μy) ./ σy
                offset = -min(minimum(zx), minimum(zy)) + 1.0
            end
            toz(v, k) = k == 1 ? (v - μx) / σx + offset : (v - μy) / σy + offset
            tor(z, k) = k == 1 ? (z - offset) * σx + μx : (z - offset) * σy + μy
            V = toz.(xt, 1); R = toz.(yt, 2)
            θ = fit_solver(m, t, V, R)
            tg = collect(range(0.0, t[end], length=400))
            y = osolve(m, θ, [V[1], R[1]], (0.0, t[end]), tg)
            if y === nothing
                @printf("✗ %s no integrable\n", m.name); continue
            end
            yr0 = decyr(d[1])
            rmse_p = sqrt(mean((tor.(osolve(m, θ, [V[1], R[1]], (0.0, t[end]), t)[1, :], 1) .- xt) .^ 2))
            for j in 1:length(tg)
                @printf(io, "%s,%.4f,%.4f,%.4f\n", m.name, yr0 + tg[j], tor(y[1, j], 1), tor(y[2, j], 2))
            end
            @printf("✓ %s ajustado en %s–%s | RMSE profits=%.2f | θ=[%s]\n",
                    m.name, year(WIN_LO), year(WIN_HI), rmse_p,
                    join((@sprintf("%.3f", v) for v in θ), ","))
        end
    end
    println("✓ $(joinpath(OUTDIR, "julia_oscillation.csv"))")
end

main_osc()
