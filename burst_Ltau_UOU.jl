using LinearAlgebra, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter, Statistics

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
    
    # Energy of the state psi
    exact_E = real(inner(psi', H, psi) / inner(psi', Id, psi))
    
    # Energy at infinite temperature (beta = 0)
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

        # Choose the stopping condition according to the search direction
        if (dbeta > 0 && current_E <= exact_E) || (dbeta < 0 && current_E >= exact_E)
            break
        end
    end
    
    return beta, current_O
end

function block_correlation_length_TM(psi::MPS, start::Int; block_size::Int = 2)
    N = length(psi)
    stop = start + block_size - 1
    if start < 1 || stop > N
        error("Block [$start, $stop] is out of range for N = $N")
    end

    psi_loc = copy(psi)
    
    # Important: Adjust the gauge so that the target block consists of orthogonal tensors without singular values
    # If start > 1, set the left neighbor as the orthogonality center (the target becomes right-orthogonal)
    # If start == 1, set the right neighbor as the orthogonality center (the target becomes left-orthogonal)
    if start > 1
        orthogonalize!(psi_loc, start - 1)
    else
        orthogonalize!(psi_loc, stop + 1)
    end

    # Construct the block tensor
    B = psi_loc[start]
    for j in (start + 1):stop
        B *= psi_loc[j]
    end

    # Obtain the left and right bond indices
    l_ind = start > 1 ? linkind(psi_loc, start - 1) : nothing
    r_ind = stop < N ? linkind(psi_loc, stop) : nothing

    if isnothing(l_ind) || isnothing(r_ind)
        return NaN, NaN, NaN
    end

    Dl = dim(l_ind)
    Dr = dim(r_ind)

    if Dl != Dr
        return NaN, NaN, NaN
    elseif Dl == 1
        # Treat the correlation length as zero when there is no entanglement (bond dimension 1)
        return 0.0, 1.0, 0.0
    end
    D = Dl

    # Obtain the physical indices in the block
    s_inds = [siteind(psi_loc, j) for j in start:stop]
    dphys = prod(dim(s) for s in s_inds)

    # Convert the ITensor to a multidimensional array and reshape it appropriately
    inds = vcat(s_inds, [l_ind, r_ind])
    Araw = Array(permute(B, inds...), inds...)
    A3 = reshape(Araw, dphys, D, D)

    # Construct the transfer matrix E using the Kronecker product
    E = zeros(ComplexF64, D^2, D^2)
    for s in 1:dphys
        As = @view A3[s, :, :]
        E .+= kron(Matrix(As), conj(Matrix(As)))
    end

    # Compute the eigenvalues
    evals = eigvals(E)
    sort!(evals, by = abs, rev = true)

    λ1 = abs(evals[1])
    λ2 = length(evals) > 1 ? abs(evals[2]) : λ1

    # Compute the correlation length
    if isapprox(λ2, λ1; rtol=1e-12, atol=1e-14)
        xi = Inf
    else
        xi = -block_size / log(λ2 / λ1)
    end

    return xi, λ1, λ2
end

function bulk_correlation_length_TM(
    psi::MPS;
    exclude_ends::Int = 4,
    block_size::Int = 2,
    statistic::Symbol = :median,
    trim_fraction::Float64 = 0.1,
    sites::Union{Nothing,AbstractVector{Int}} = nothing,
)
    N = length(psi)

    # Block starting positions to use
    starts = if sites === nothing
        lo = exclude_ends + 1
        hi = N - exclude_ends - block_size + 1
        collect(lo:hi)
    else
        collect(sites)
    end

    if isempty(starts)
        error("No bulk blocks selected. Decrease exclude_ends or block_size.")
    end

    xis = Float64[]
    lambdas1 = Float64[]
    lambdas2 = Float64[]
    used_starts = Int[]

    for i in starts
        xi, λ1, λ2 = block_correlation_length_TM(psi, i; block_size = block_size)
        if isfinite(xi) && !isnan(xi)
            push!(xis, xi)
            push!(lambdas1, λ1)
            push!(lambdas2, λ2)
            push!(used_starts, i)
        end
    end

    if isempty(xis)
        error("No usable bulk blocks found. Try smaller exclude_ends or check bond dimensions.")
    end

    xi_bulk = if statistic == :median
        median(xis)
    elseif statistic == :mean
        mean(xis)
    elseif statistic == :trimmedmean
        ys = sort(xis)
        n = length(ys)
        k = floor(Int, trim_fraction * n)
        if 2k >= n
            mean(ys)
        else
            mean(view(ys, k+1:n-k))
        end
    else
        error("statistic must be :median, :mean, or :trimmedmean")
    end

    return (
        xi_bulk = xi_bulk,
        xi_local = xis,
        lambda1_local = lambdas1,
        lambda2_local = lambdas2,
        block_starts = used_starts,
        block_size = block_size,
        statistic = statistic,
    )
end

function estimate_correlation_length(psi::MPS, direction::String; threshold::Float64=1e-7)
    # Convert inputs "x", "y", and "z" to "Sx", "Sy", and "Sz"
    spin = length(direction) == 1 ? "S" * lowercase(direction) : direction
    
    # Explicitly convert to complex to avoid ITensors issues and support complex states
    psi_c = complex(psi)
    L = length(psi_c)
    
    # Compute expectation values and the correlation matrix
    spin_expect = expect(psi_c, spin)
    corr_matrix = correlation_matrix(psi_c, spin, spin)
    
    # Connected correlation: <S_i S_j> - <S_i><S_j>
    conn_corr = corr_matrix - spin_expect * spin_expect'
    
    # Obtain the correlation decay as a function of distance r to the right of the center site
    center = div(L, 2)
    rs = 1:(L - center)
    c_vals = [abs(conn_corr[center, center + r]) for r in rs]
    
    # Use only data points larger than the specified threshold for fitting
    valid_idx = findall(x -> x > threshold, c_vals)
    r_valid = rs[valid_idx]
    c_valid = c_vals[valid_idx]
    
    n_valid = length(r_valid)
    xi = NaN
    std_xi = NaN
    
    if n_valid >= 2
        # Linear fit of log(C(r)) = -r/xi + log(A)
        y = log.(c_valid)
        X = hcat(r_valid, ones(n_valid))
        params = X \ y  
        a = params[1] # Slope -1/xi
        
        # Compute the correlation length only when the slope is negative (decaying)
        if a < 0
            xi = -1 / a
            
            # Also compute the standard error when there are at least three data points
            if n_valid >= 3
                y_fit = X * params                   
                rss = sum((y .- y_fit).^2)           # Residual sum of squares
                s2 = rss / (n_valid - 2)             # Residual variance
                
                r_mean = sum(r_valid) / n_valid
                var_a = s2 / sum((r_valid .- r_mean).^2) 
                std_a = sqrt(var_a)                  # Standard error of the slope
                
                # Error propagation: |dx/da| * std_a = (1/a^2) * std_a
                std_xi = std_a / (a^2)               
            end
        end
    end
    
    return xi, std_xi
end

function main()

    Ls = [20, 30, 40]
    chi = 10
    dt = 0.2
    rep = 150
    ttotal = dt * rep
    maxdim_obs = 2048
    maxdim_state = 128
    trunc = 1e-7
    Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045/2, 0.0, 0.8090/2
    beta = 0.1
    penalty_coeff = 0.0
    observable = "Magz"
    num_parts = 30
    
    # Added: Number of trials
    num_trials = 3

    burst_results = Dict{Int, Vector{Float64}}()
    eq_results = Dict{Int, Vector{Float64}}()   # current_O
    dyn_results = Dict{Int, Vector{Float64}}()  # -real(inner(...))
    xi_results_TM = Dict{Int, Vector{Float64}}()
    xi_results_x = Dict{Int, Vector{Float64}}()
    std_xi_results_x = Dict{Int, Vector{Float64}}()
    xi_results_y = Dict{Int, Vector{Float64}}()
    std_xi_results_y = Dict{Int, Vector{Float64}}()
    xi_results_z = Dict{Int, Vector{Float64}}()
    std_xi_results_z = Dict{Int, Vector{Float64}}()

    local ts

    for L in Ls
        lambda = penalty_coeff/L^2

        # --- Reconstruct the filename ---
        local cache_file
        cache_file = "submit_UOU/$(observable)_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"

        # --- Load the data ---
        if isfile(cache_file)
            # println("File found: $cache_file")
            data = load(cache_file)
            s = data["s"]
            ts = data["ts"]
            O_Us = data["O_Us"]
            # println("Finished loading data.")
        else
            println("File not found: $cache_file")
            return # Stop if the file is missing
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
        xi_values_TM = Float64[]
        xi_values_x = Float64[]
        std_xi_values_x = Float64[]
        xi_values_y = Float64[]
        std_xi_values_y = Float64[]
        xi_values_z = Float64[]
        std_xi_values_z = Float64[]

        @showprogress "Calculating L=$L" for i in 1:(num_parts+1)
            tau = ts[i]
            O_tau = O_Us[i]

            current_max_burst = -Inf # Variable holding the maximum value
            best_eq_val = 0.0        # Equilibrium value at the maximum
            best_dyn_val = 0.0       # Dynamical value at the maximum
            best_xi_TM = NaN            # Added: xi at the maximum
            best_xi_x = NaN             # Added: xi_x at the maximum
            best_std_xi_x = NaN        # Added: std_xi_x at the maximum
            best_xi_y = NaN             # Added: xi_y at the maximum
            best_std_xi_y = NaN        # Added: std_xi_y at the maximum
            best_xi_z = NaN             # Added: xi_z at the maximum
            best_std_xi_z = NaN        # Added: std_xi_z at the maximum
            

            # Trial loop
            for trial in 1:num_trials
                
                # Change seed for each trial (Depend on L, i, trial for reproducibility)

                if trial != num_trials
                    Random.seed!(1000 + L*100 + i*10 + trial)
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
                
                xi_TM = bulk_correlation_length_TM(psi; exclude_ends=5, block_size=2, statistic=:median).xi_bulk
                xi_x, std_xi_x = estimate_correlation_length(psi, "Sx"; threshold=1e-7)
                xi_y, std_xi_y = estimate_correlation_length(psi, "Sy"; threshold=1e-7)
                xi_z, std_xi_z = estimate_correlation_length(psi, "Sz"; threshold=1e-7)

                # Inverse temperature and equilibrium value
                beta_true, current_O = calc_beta_and_obs(L, s, psi, O, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
                
                # Compute the result of the current trial
                dyn_val = -real(inner(psi_evolved', O, psi_evolved)) # Dynamical contribution
                val = current_O + dyn_val # Burst value
                
                # Update the maximum value
                if val > current_max_burst
                    current_max_burst = val
                    best_eq_val = current_O
                    best_dyn_val = dyn_val
                    best_xi_TM = xi_TM           # Added: Save xi at the maximum
                    best_xi_x = xi_x             # Added: Save xi_x at the maximum
                    best_std_xi_x = std_xi_x     # Added: Save std_xi_x at the maximum
                    best_xi_y = xi_y             # Added: Save xi_y at the maximum
                    best_std_xi_y = std_xi_y     # Added: Save std_xi_y at the maximum
                    best_xi_z = xi_z             # Added: Save xi_z at the maximum
                    best_std_xi_z = std_xi_z     # Added: Save std_xi_z at the maximum
                end
            end
            
            # Save the maximum value
            push!(burst_values, current_max_burst)
            push!(eq_values, best_eq_val)
            push!(dyn_values, best_dyn_val)
            push!(xi_values_TM, best_xi_TM)      # Fixed: Push best_xi
            push!(xi_values_x, best_xi_x)        # Fixed: Push best_xi
            push!(std_xi_values_x, best_std_xi_x)# Fixed: Push best_std_xi
            push!(xi_values_y, best_xi_y)        # Fixed: Push best_xi
            push!(std_xi_values_y, best_std_xi_y)# Fixed: Push best_std_xi
            push!(xi_values_z, best_xi_z)        # Fixed: Push best_xi
            push!(std_xi_values_z, best_std_xi_z)# Fixed: Push best_std_xi
        end

        burst_results[L] = burst_values
        eq_results[L] = eq_values
        dyn_results[L] = dyn_values
        xi_results_TM[L] = xi_values_TM
        xi_results_x[L] = xi_values_x
        std_xi_results_x[L] = std_xi_values_x
        xi_results_y[L] = xi_values_y
        std_xi_results_y[L] = std_xi_values_y
        xi_results_z[L] = xi_values_z
        std_xi_results_z[L] = std_xi_values_z

        # println("L = $L:")
        # println("burst_values: ", burst_values)
        # println("xi_values_TM: ", xi_values_TM)
        # println("xi_values_x: ", xi_values_x)
        # println("std_xi_values_x: ", std_xi_values_x)
        # println("xi_values_y: ", xi_values_y)
        # println("std_xi_values_y: ", std_xi_values_y)
        # println("xi_values_z: ", xi_values_z)
        # println("std_xi_values_z: ", std_xi_values_z)
    end

    results_filename = "Ltauvsburst_UOU_$(observable)_T$(ttotal)_lambda$(penalty_coeff)_chi$(chi).jld2"
    println("Saving results to $results_filename...")
    save(results_filename, "plot_ts", ts, "burst_results", burst_results, "eq_results", eq_results, "dyn_results", dyn_results, "xi_values_TM", xi_results_TM, "xi_values_x", xi_results_x, "std_xi_values_x", std_xi_results_x, "xi_values_y", xi_results_y, "std_xi_values_y", std_xi_results_y, "xi_values_z", xi_results_z, "std_xi_values_z", std_xi_results_z)
    println("Finished saving.")
    
end

main()
