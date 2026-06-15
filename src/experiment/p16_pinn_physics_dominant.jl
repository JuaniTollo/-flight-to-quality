# ============================================================
# p16_pinn_physics_dominant.jl — PINN INVERSO: BARRIDO DE CAPACIDAD con física regularizadora
# ============================================================
# Inverso diferenciable del núcleo físico (cadena de maduración → kernel Gamma; se aprenden μ,k).
# Red sustituta t→[P,I] con features de Fourier; pérdida L = L_data + λ·L_phys + λ_reg·‖θ_red‖².
#
# PREGUNTA (diseño correcto, no dos extremos): con la PÉRDIDA FÍSICA dominante de regularizador,
# ¿qué CAPACIDAD de red identifica el retardo sin sobreajustar? Barremos el ancho de la red
# (≈80 → ≈2200 pesos, cruzando los ~290 datos) a λ fijo, y miramos la curva sesgo-varianza:
# R² de ENTRENAMIENTO vs R² de VALIDACIÓN (1 de cada 5 puntos reservado). Si coinciden → no hay
# sobreajuste (la física regulariza); si R²_valid se hunde al crecer la red → sí sobreajusta.
# El μ̂ por capacidad dice si el retardo se ancla en algún punto.
#
# Transform: log trimestre a trimestre. Validación sintética (lag conocido) para la mayor capacidad.
# Corre:  julia --project=julia src/experiment/p16_pinn_physics_dominant.jl   [ENV EPOCHS, LAMBDA]
# ============================================================
using Lux, Optimisers, Zygote, Random, Statistics, Printf, DelimitedFiles
using Plots

const SHAPE     = 4.0
const KMAX      = 10
const N_FOURIER = 6
const λ_reg     = 1e-3       # weight decay leve; el regularizador principal es la física (λ)
const λ_PHYS    = parse(Float64, get(ENV, "LAMBDA", "10"))   # física dominante (regulariza la red)
const VAL_EVERY = 5
const REPO      = abspath(joinpath(@__DIR__, "..", ".."))
const CSVP      = joinpath(REPO, "data", "processed", "lotka_volterra.csv")
const OUTDIR    = joinpath(REPO, "output", "experiment"); mkpath(OUTDIR)

gamma_kernel(μ) = (scale=μ/SHAPE; xs=collect(0:KMAX).+0.5; w=(xs.^(SHAPE-1)).*exp.(-xs./scale); w./sum(w))

# ---------------------------------------------------------------- datos sintéticos (señal fuerte, lag conocido)
function synth_data(; μ_true_q=4.0, T=24.0, dt=0.25, noise=0.03, seed=0)
    θ=(μ_true_q/4)/SHAPE; a,b,c,d,k=0.9,0.4,0.6,0.4,1.1; nstep=round(Int,T/dt)
    u=[1.0,0.5,0.0,0.0,0.0,0.0]; P=Float64[]; I=Float64[]
    for s in 0:nstep
        push!(P,u[1]); push!(I,u[2]); P_,I_,m1,m2,m3,m4=u
        du=[c*I_-d*P_-k*m4, a*P_-b*I_, (I_-m1)/θ, (m1-m2)/θ, (m2-m3)/θ, (m3-m4)/θ]
        u=u.+dt.*du
    end
    rng=MersenneTwister(seed)
    P.+=noise*std(P).*randn(rng,length(P)); I.+=noise*std(I).*randn(rng,length(I))
    z(v)=(v.-mean(v))./std(v); return z(P), z(I)
end

# ---------------------------------------------------------------- red (ancho h parametrizable)
make_freqs(nfreq,fmax)=collect(exp.(range(log(1.0),log(fmax);length=nfreq)))
function fourier_features(t,t0,T,freqs)
    τ=(t[1]-t0)/(T-t0); return vcat([[sin(2π*f*τ),cos(2π*f*τ)] for f in freqs]...)
end
function build_nn(t0,T,oscale,h; nfreq=N_FOURIER, fmax=20.0)
    freqs=make_freqs(nfreq,fmax)
    Chain(WrappedFunction(t->fourier_features(t,t0,T,freqs)),
          Dense(2*nfreq=>h,tanh), Dense(h=>h,tanh), Dense(h=>2),
          WrappedFunction(x->oscale.*x))
end
nparams(nn)=Int(Lux.parameterlength(Lux.setup(MersenneTwister(0),nn)[1]))
nn_dudt(nn,t,ps,st;h=1e-4)=(nn([t+h],ps,st)[1].-nn([t-h],ps,st)[1])./(2h)

_l2(x::AbstractArray)=sum(abs2,x); _l2(x::Number)=abs2(x)
_l2(x::NamedTuple)=isempty(x) ? 0.0 : sum(_l2,values(x))
_l2(x::Tuple)=isempty(x) ? 0.0 : sum(_l2,x); _l2(x)=0.0

# ---------------------------------------------------------------- entrenamiento (ancho h, λ_phys)
function train(t,PI_obs,p_init,oscale,h,λ_phys; tidx=collect(1:length(t)), vidx=Int[], n_epochs=4000, seed=42, lr=1e-2, fmax=20.0, evalevery=50, patience=8)
    n=length(t); nn=build_nn(t[1],t[end],oscale,h;fmax=fmax)
    nn_ps,st=Lux.setup(MersenneTwister(seed),nn); st=Lux.testmode(st); nn_ps=Lux.f64(nn_ps)
    theta=(nn=nn_ps, ode=copy(p_init))
    function comps(th)
        ps=th.nn; a,b,c,d,k=th.ode[1],th.ode[2],th.ode[3],th.ode[4],th.ode[5]; μ=exp(th.ode[6]); w=gamma_kernel(μ)
        P=[nn([t[i]],ps,st)[1][1] for i in 1:n]; I=[nn([t[i]],ps,st)[1][2] for i in 1:n]
        data=mean([(P[i]-PI_obs[1,i])^2+(I[i]-PI_obs[2,i])^2 for i in tidx])
        m=[sum(w[j+1]*I[i-j] for j in 0:min(KMAX,i-1)) for i in 1:n]
        phys=mean(map((KMAX+1):(n-1)) do i
            du=nn_dudt(nn,t[i],ps,st); rP=du[1]-(c*I[i]-d*P[i]-k*m[i]); rI=du[2]-(a*P[i]-b*I[i]); rP^2+rI^2
        end)
        return data, phys
    end
    loss(th)=(let (dd,pp)=comps(th); dd+λ_phys*pp+λ_reg*_l2(th.nn) end)
    # pérdida de DATOS en validación (para early stopping)
    valloss(th)=(isempty(vidx) ? 0.0 :
        mean([(nn([t[i]],th.nn,st)[1][1]-PI_obs[1,i])^2+(nn([t[i]],th.nn,st)[1][2]-PI_obs[2,i])^2 for i in vidx]))
    opt=Optimisers.setup(Adam(lr),theta)
    best_val=Inf; best=deepcopy(theta); nobad=0
    for epoch in 1:n_epochs
        _,g=Zygote.withgradient(loss,theta); opt,theta=Optimisers.update(opt,theta,g[1])
        if !isempty(vidx) && epoch % evalevery == 0     # EARLY STOPPING por validación
            vl=valloss(theta)
            if vl < best_val; best_val=vl; best=deepcopy(theta); nobad=0
            else; nobad+=1; nobad>=patience && break; end
        end
    end
    return (isempty(vidx) ? theta : best), nn, st       # devuelve el MEJOR checkpoint de validación
end

function logtrim_window(; from_year=1990)
    raw,h=readdlm(CSVP,',',header=true); hv=vec(h)
    di=findfirst(==("DATE"),hv); pi_=findfirst(==("PROFITS"),hv); ii=findfirst(==("INVESTMENT"),hv)
    rows=[r for r in 1:size(raw,1) if raw[r,pi_]!="" && raw[r,ii]!=""]; dts=[string(raw[r,di]) for r in rows]
    Pl=Float64[raw[r,pi_] for r in rows]; Il=Float64[raw[r,ii] for r in rows]
    Plt=100 .*(log.(Pl[2:end]).-log.(Pl[1:end-1])); Ilt=100 .*(log.(Il[2:end]).-log.(Il[1:end-1])); dq=dts[2:end]
    keep=[parse(Int,d[1:4])>=from_year for d in dq]
    yrs=[parse(Int,d[1:4]) for d in dq[keep]]
    return Plt[keep], Ilt[keep], yrs        # CRUDO: el z-score se ajusta en train (anti-leakage)
end

function main(; n_epochs=4000, from_year=1990)
    println("="^74); println("PINN INVERSO — BARRIDO DE CAPACIDAD (física dominante λ=$(λ_PHYS) regulariza)  [log-trim]"); println("="^74)
    P,I,yrs=logtrim_window(from_year=from_year); n=length(P)
    # validación ESTRATÉGICA: bloque contiguo de la recesión ESTÁNDAR de 2001 (dot-com), una crisis
    # típica (no el outlier de 2008/2020). Testea generalización a una crisis NO vista (forecast).
    vidx=findall(y->2000<=y<=2002, yrs); tidx=setdiff(collect(1:n),vidx)
    # z-score con estadísticos de ENTRENAMIENTO solamente (anti-leakage; el scaler no ve la validación)
    mP,sP=mean(P[tidx]),std(P[tidx]); mI,sI=mean(I[tidx]),std(I[tidx]); P=(P.-mP)./sP; I=(I.-mI)./sI
    t=collect(0.0:0.25:0.25*(n-1)); PI=permutedims(hcat(P,I)); oscale=maximum(abs.(PI)); fmax=n/2.0
    r2of(idx,Pp,Ip)=1-(sum((P[idx].-Pp[idx]).^2)+sum((I[idx].-Ip[idx]).^2))/(sum((P[idx].-mean(P[idx])).^2)+sum((I[idx].-mean(I[idx])).^2))
    p_init=[0.5,0.3,0.5,0.3,0.6,log(6.0)]
    @printf("datos=%d×2=%d │ entren=%d / valid=%d (bloque 2000–2002, recesión de 2001) │ %d épocas\n", n,2n,length(tidx),length(vidx),n_epochs)

    # grilla chica (lo que vimos en clase): λ_phys (datos→física) × pesos (red chica vs grande)
    λs=[0.3,1.0,3.0,10.0,30.0,100.0,300.0]; hs=[8,24]   # λ fino (log) × 2 tamaños de red
    np8=nparams(build_nn(t[1],t[end],oscale,8;fmax=fmax)); np24=nparams(build_nn(t[1],t[end],oscale,24;fmax=fmax))
    @printf("redes: chica h=8 → %d pesos (%.2f× datos) │ grande h=24 → %d pesos (%.2f× datos)\n\n", np8,np8/(2n),np24,np24/(2n))
    grid=Dict{Tuple{Float64,Int},NamedTuple}()
    @printf("%-8s %-8s %-8s %-10s %-10s\n","λ_phys","pesos","μ̂","R²entren","R²valid")
    for λ in λs, h in hs
        np = h==8 ? np8 : np24
        theta,nn,st=train(t,PI,p_init,oscale,h,λ; tidx=tidx,vidx=vidx,n_epochs=n_epochs,fmax=fmax)
        μ̂=exp(theta.ode[6]); Pp=[nn([τ],theta.nn,st)[1][1] for τ in t]; Ip=[nn([τ],theta.nn,st)[1][2] for τ in t]
        r2tr=r2of(tidx,Pp,Ip); r2va=r2of(vidx,Pp,Ip)
        grid[(λ,h)]=(np=np,μ=μ̂,r2tr=r2tr,r2va=r2va); flush(stdout)
        @printf("%-8.3g %-8d %-8.2f %-10.3f %-10.3f\n",λ,np,μ̂,r2tr,r2va)
    end

    # validación sintética (señal fuerte): ¿recupera un lag conocido? (red grande, física dominante)
    Ps,Is=synth_data(μ_true_q=4.0); ns=length(Ps); tsy=collect(0.0:0.25:0.25*(ns-1)); PIs=permutedims(hcat(Ps,Is))
    th_s,_,_=train(tsy,PIs,[0.5,0.2,0.3,0.2,0.5,log(8.0)],maximum(abs.(PIs)),24,10.0; n_epochs=n_epochs,fmax=ns/2.0)
    μsyn=exp(th_s.ode[6])
    @printf("\nSINTÉTICO (señal fuerte, h=24, λ=10): verdadero=4.0 → recuperado=%.2f trim (error %.0f%%)\n", μsyn,100*abs(μsyn-4)/4)

    # ---- figura: R²(valid y entren) y μ̂ vs λ, una serie por tamaño de red ----
    g(λ,h,f)=getfield(grid[(λ,h)],f)
    p1=plot(λs,[g(λ,8,:r2va) for λ in λs],xscale=:log10,marker=:circle,lw=2,c=:seagreen,label="valid · chica",
            xlabel="λ_phys (peso de la física)",ylabel="R²(datos)",title="Generalización",legend=:bottomright)
    plot!(p1,λs,[g(λ,24,:r2va) for λ in λs],xscale=:log10,marker=:diamond,lw=2,c=:darkorange,label="valid · grande")
    plot!(p1,λs,[g(λ,8,:r2tr) for λ in λs],xscale=:log10,ls=:dash,lw=1,c=:seagreen,label="entren · chica")
    plot!(p1,λs,[g(λ,24,:r2tr) for λ in λs],xscale=:log10,ls=:dash,lw=1,c=:darkorange,label="entren · grande")
    p2=plot(λs,[g(λ,8,:μ) for λ in λs],xscale=:log10,marker=:circle,lw=2,c=:seagreen,label="red chica",
            xlabel="λ_phys",ylabel="μ̂ (trim)",title="¿Se ancla el retardo?",legend=:topright)
    plot!(p2,λs,[g(λ,24,:μ) for λ in λs],xscale=:log10,marker=:diamond,lw=2,c=:darkorange,label="red grande")
    hline!(p2,[5.0],ls=:dash,c=:gray,label="~1 año (clásicos)")
    fig=plot(p1,p2,layout=(1,2),size=(1150,450),plot_title="PINN inverso: λ × capacidad de red (log-trim)")
    savefig(fig,joinpath(OUTDIR,"p16_pinn_physics.pdf")); savefig(fig,joinpath(OUTDIR,"p16_pinn_physics.png")); println("\n✓ figura: ",joinpath(OUTDIR,"p16_pinn_physics.pdf"))
    open(joinpath(REPO,"results","p16_pinn_physics.csv"),"w") do io
        println(io,"lambda,pesos,mu_trim,r2_train,r2_valid")
        for λ in λs, h in hs; r=grid[(λ,h)]; @printf(io,"%.3g,%d,%.4f,%.4f,%.4f\n",λ,r.np,r.μ,r.r2tr,r.r2va); end
    end
    println("✓ csv"); return grid
end

main(n_epochs = parse(Int, get(ENV, "EPOCHS", "4000")))
