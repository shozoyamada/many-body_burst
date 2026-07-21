using LinearAlgebra, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter

# Functions

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

function Eq_beta(N, s, beta, O, Jx, Jy, Jz, hx, hy, hz; dbeta = 0.0001, maxdim=2048, cutoff=1e-14)
    Id = MPO(s, n -> "Id")
    rho = copy(Id)
    gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)
    steps = round(Int, abs(beta / dbeta))
    for step in 1:steps
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
    end
    tr_rho = inner(Id, rho)
    eq_O = real(inner(O, rho) / tr_rho)
    return eq_O
end

function calc_beta_and_obs(N, s, psi, O, Jx, Jy, Jz, hx, hy, hz; dbeta_abs=0.0001, max_steps=10000, maxdim=2048, cutoff=1e-14)

    Id = MPO(s, n -> "Id")
    H = Heisenberg(N, s, Jx, Jy, Jz, hx, hy, hz)
    
    # Energy of state psi
    exact_E = real(inner(psi', H, psi) / inner(psi', Id, psi))
    
    # Energy at infinite temperature (beta=0)
    rho = copy(Id)
    tr_rho_inf = inner(Id, rho)
    E_inf = real(inner(H, rho) / tr_rho_inf)

    # Determine the beta search direction (positive or negative) from the energy ordering
    if exact_E < E_inf
        dbeta = dbeta_abs
    else
        dbeta = -dbeta_abs
    end

    beta = 0.0
    current_O = 0.0
    
    # Imaginary-time evolution operator
    gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)

    for step in 1:max_steps
        beta += dbeta
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
        tr_rho = inner(Id, rho)
        current_E = real(inner(H, rho) / tr_rho)
        current_O = real(inner(O, rho) / tr_rho)

        # Branch the stopping condition according to the search direction
        if (dbeta > 0 && current_E <= exact_E) || (dbeta < 0 && current_E >= exact_E)
            break
        end
    end
    
    return beta, current_O
end

function find_cache_file(cache_dir::String, observable::String, L::Int, dt, ttotal, maxdim_obs::Int, num_parts::Int, target_hz::Float64; atol::Float64=1e-8)
    pattern = Regex("^$(observable)_Ising_hz([0-9eE+\\-.]+)_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts)\\.jld2\$")
    candidates = Tuple{Float64, String}[]

    isdir(cache_dir) || return nothing

    for entry in readdir(cache_dir)
        m = match(pattern, entry)
        if m !== nothing
            parsed_hz = tryparse(Float64, m.captures[1])
            if parsed_hz !== nothing
                push!(candidates, (parsed_hz, joinpath(cache_dir, entry)))
            end
        end
    end

    if isempty(candidates)
        return nothing
    end

    best_hz, best_path = candidates[1]
    best_dist = abs(best_hz - target_hz)
    for (candidate_hz, candidate_path) in candidates[2:end]
        candidate_dist = abs(candidate_hz - target_hz)
        if candidate_dist < best_dist
            best_hz = candidate_hz
            best_path = candidate_path
            best_dist = candidate_dist
        end
    end

    return best_dist <= atol ? best_path : nothing
end

function main()

    L = 40
    chi = 10
    dt = 0.2
    rep = 150
    ttotal = dt * rep
    maxdim_obs = 2048
    maxdim_state = 128
    trunc = 1e-7
    Jx, Jy, Jz, hx, hy = 0.0, 0.0, 1.0, 0.9045/2, 0.0
    hzs = 0.0:0.04045:0.4045
    beta = 0.1
    penalty_coeff = 72.0
    observable = "Magy"
    num_parts = 30
    
    # Added: number of trials
    num_trials = 3

    burst_results = Dict{Float64, Vector{Float64}}()
    eq_results = Dict{Float64, Vector{Float64}}()   # current_O
    dyn_results = Dict{Float64, Vector{Float64}}()  # -real(inner(...))

    local ts

    for (hz_idx, hz_raw) in enumerate(hzs)
        hz = Float64(hz_raw)
        hz_key = round(hz, digits=6)
        lambda = penalty_coeff/L^2

        # --- Reconstruct the filename (allow nearby hz values) ---
        local cache_file
        cache_file = find_cache_file("submit_hz_UOU", observable, L, dt, ttotal, maxdim_obs, num_parts, hz)
        if cache_file === nothing
            cache_file = "submit_hz_UOU/$(observable)_Ising_hz$(hz)_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"
        end

        # --- Load data ---
        if isfile(cache_file)
            # println("File found: $cache_file")
            data = load(cache_file)
            s = data["s"]
            ts = data["ts"]
            O_Us = data["O_Us"]
            # println("Finished loading data.")
        else
            println("File not found: $cache_file")
            return # Exit if the file is missing
        end

        local O
        if observable == "Szc"
            c = div(L, 2) + 1 # Center site
            os_Szc = OpSum()
            os_Szc += "Sz", c
            O = MPO(os_Szc,s) # Sz at site c
        elseif observable == "Magy"
            O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 1/L, 0.0)
        elseif observable == "Magz"
            O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 0.0, 1/L)
        end

        gate = Trotter(L, s, Jx, Jy, Jz, hx, hy, hz, dt)
        gate_dag = Trotter(L, s, Jx, Jy, Jz, hx, hy, hz, -dt)
        H_phys = Heisenberg(L, s, Jx, Jy, Jz, hx, hy, hz)
        Id = MPO(s, n -> "Id")
        E_target = Eq_beta(L, s, beta, H_phys, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
        H_diff = H_phys - E_target * Id
        
        H_penalty = lambda * apply(H_diff, H_diff; maxdim=maxdim_obs, cutoff=trunc)

        burst_values = Float64[]
        eq_values = Float64[]
        dyn_values = Float64[]

        @showprogress "Calculating hz=$hz_key" for i in 1:(num_parts+1)
            tau = ts[i]
            O_tau = O_Us[i]

            current_max_burst = -Inf # Variable that stores the maximum value
            best_eq_val = 0.0        # Equilibrium value at the maximum
            best_dyn_val = 0.0       # Dynamical value at the maximum

            # Trial loop
            for trial in 1:num_trials
                
                # Change seed for each trial (Depend on hz, i, trial for reproducibility)

                if trial != num_trials
                    Random.seed!(1000 + hz_idx*1000 + i*10 + trial)
                    psi0 = random_mps(ComplexF64, s, linkdims = 1)
                
                else
                    if observable == "Magy"
                        psi0 = MPS(ComplexF64, s, n -> "Y-")
                    else
                        psi0 = MPS(ComplexF64, s, n -> "Dn")
                    end
                    for j in 1:(i-1)*div(rep, num_parts)
                        psi0 = apply(gate_dag, psi0; maxdim=maxdim_state, cutoff=trunc)
                        normalize!(psi0)
                    end
                end

                nsweeps = 50
                # Fix: Match array length to nsweeps
                maxdim = fill(chi, nsweeps)
                noise_schedule = [1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10, 1e-11, 1e-11, 1e-12, 1e-12, 1e-13, 1e-13, 1e-14, 1e-14, 0.0]

                eigs, psi = dmrg([O_tau, H_penalty], psi0; nsweeps=nsweeps, maxdim=maxdim, noise=noise_schedule, outputlevel=0)

                psi_evolved = psi
                for j in 1:(i-1)*div(rep, num_parts)
                    psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
                    normalize!(psi_evolved)
                end

                # Inverse temperature and equilibrium value
                _, current_O = calc_beta_and_obs(L, s, psi, O, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)

                # Calculate the result of the current trial
                dyn_val = -real(inner(psi_evolved', O, psi_evolved)) # Dynamical contribution
                val = current_O + dyn_val # Burst value
                
                # Update the maximum value
                if val > current_max_burst
                    current_max_burst = val
                    best_eq_val = current_O
                    best_dyn_val = dyn_val
                end
            end
            
            # Store the maximum value
            push!(burst_values, current_max_burst)
            push!(eq_values, best_eq_val)
            push!(dyn_values, best_dyn_val)
        end

        burst_results[hz_key] = burst_values
        eq_results[hz_key] = eq_values
        dyn_results[hz_key] = dyn_values

        println("hz=$hz_key burst_results=$(burst_results[hz_key])")
        flush(stdout)
    end

    results_filename = "hztauvsburst_UOU_$(observable)_T$(ttotal)_lambda$(penalty_coeff)_chi$(chi)_trials$(num_trials).jld2"
    println("Saving results to $results_filename...")
    save(results_filename, "hzs", hzs, "plot_ts", ts, "burst_results", burst_results, "eq_results", eq_results, "dyn_results", dyn_results)
    println("Finished saving.")
    flush(stdout)
    
end

main()
