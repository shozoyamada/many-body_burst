using LinearAlgebra, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter

# functions

function Heisenberg(N, s, Jx, Jy, Jz, hx, hy, hz)
    os_phys = OpSum()
    for j in 1:(N - 1)
        os_phys += Jx, "Sx", j, "Sx", j + 1
        os_phys += Jy, "Sy", j, "Sy", j + 1
        os_phys += Jz, "Sz", j, "Sz", j + 1
        os_phys += hx, "Sx", j
        os_phys += hy, "Sy", j
        os_phys += hz, "Sz", j
    end
    os_phys += hx, "Sx", N
    os_phys += hy, "Sy", N
    os_phys += hz, "Sz", N
    return MPO(os_phys,s)
end

function Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, dt)
    # second-order Trotter decomposition
    gate = ITensor[]

    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj =
        Jx * op("Sx", s1) * op("Sx", s2) +
        Jy * op("Sy", s1) * op("Sy", s2) +
        Jz * op("Sz", s1) * op("Sz", s2) +
        hx * op("Sx", s1) * op("Id", s2) +
        hy * op("Sy", s1) * op("Id", s2) +
        hz * op("Sz", s1) * op("Id", s2)
        Gj = exp(-im * dt / 2 * hj)
        push!(gate, Gj)
    end
    hN = hx * op("Sx", s[N]) + hy * op("Sy", s[N]) + hz * op("Sz", s[N])
    GN = exp(-im * dt / 2 * hN)
    push!(gate, GN)
    append!(gate, reverse(gate))

    return gate
end

function parts(O_initial::MPO, gate::Vector{ITensor}, dt::Float64, rep::Int, part::Int, maxdim_obs::Int, trunc::Float64)
    ts = Float64[]
    O_Us = MPO[]
    errors = Float64[]
    norm_O = norm(O_initial)
    O_t = O_initial  # time-evolved operator
    l = div(rep, part)

    push!(ts, 0.0)
    push!(O_Us, O_t)
    push!(errors, 0.0)

    @showprogress "Evolving operator in Heisenberg picture" for k in 1:rep
        # O(t+τ) = U(τ)† O(t) U(τ)

        O_t = apply(gate, O_t; apply_dag = true, maxdim=maxdim_obs, cutoff=trunc)
        err = 1-(norm(O_t) / norm_O)^2 # truncation error

        if k % l == 0
            push!(ts, k * dt)
            push!(O_Us, O_t)
            push!(errors, err)
        end
    end
    return ts, O_Us, errors
end

function main()
    # parameters definition

    L = 40
    dt = 0.2
    rep = 150
    ttotal = dt * rep
    maxdim_obs = 2048
    trunc = 1e-7

    Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045/2, 0.0, 0.8090/2

    num_parts = 30

    observable = "Magz"

    # define cache file name
    local cache_file
    if observable == "Szc"
        cache_file = "Szc_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"
    elseif observable == "Magy"
        cache_file = "Magy_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"
    elseif observable == "Magz"
        cache_file = "Magz_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"
    end

    # load or calculate ts and O_Us

    local ts, O_Us, s, errors

    if isfile(cache_file)
        println("Loading ts, O_Us, and s from $cache_file...")
        data = load(cache_file)
        ts = data["ts"]
        O_Us = data["O_Us"]
        s = data["s"]
        errors = data["errors"]
        println("Finished loading.")
    else
        println("Cache file not found. Calculating ts and O_Us...")
        s = siteinds("S=1/2", L)
        gate_dag = Trotter(L, s, Jx, Jy, Jz, hx, hy, hz, -dt)

        local O
        if observable == "Szc"
            c = div(L, 2) + 1
            os_Szc = OpSum()
            os_Szc += "Sz", c
            O = MPO(os_Szc, s)
        elseif observable == "Magy"
            O = Heisenberg(L,s,0.0,0.0,0.0,0.0,1/L,0.0)
        elseif observable == "Magz"
            O = Heisenberg(L,s,0.0,0.0,0.0,0.0,0.0,1/L)
        end
        
        ts, O_Us, errors = parts(O, gate_dag, dt, rep, num_parts, maxdim_obs, trunc)

        println(errors)
        println("Finished calculation. Saving to $cache_file...")
        save(cache_file, "s", s, "ts", ts, "O_Us", O_Us, "errors", errors)
        println("Finished saving.")
    end

end

# run main function
main()
