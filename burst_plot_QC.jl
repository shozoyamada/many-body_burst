using LinearAlgebra, Statistics, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter

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
    if beta >= 0
        gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)
    else
        gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, im * dbeta)
    end
    
    steps = round(Int, abs(beta / dbeta))
    for step in 1:steps
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
    end
    tr_rho = inner(Id, rho)
    eq_O = real(inner(O, rho) / tr_rho)
    return eq_O
end

function entanglement_entropy(psi::MPS,b::Int)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
    end
    return SvN
end

function calc_beta_and_obs(N, s, psi, O, Jx, Jy, Jz, hx, hy, hz; dbeta_abs=0.0001, max_steps=10000, maxdim=1024, cutoff=1e-14)

    Id = MPO(s, n -> "Id")
    H = Heisenberg(N, s, Jx, Jy, Jz, hx, hy, hz)
    
    # Energy of state psi
    exact_E = real(inner(psi', H, psi) / inner(psi', Id, psi))
    
    # Energy at infinite temperature (beta=0)
    rho = copy(Id)
    tr_rho_inf = inner(Id, rho)
    E_inf = real(inner(H, rho) / tr_rho_inf)

    # Determine beta search direction (positive/negative) based on energy relation
    if exact_E < E_inf
        dbeta = dbeta_abs
    else
        dbeta = -dbeta_abs
    end

    beta = 0.0
    current_O = 0.0
    
    # Imaginary time evolution operator
    gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)

    for step in 1:max_steps
        beta += dbeta
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
        tr_rho = inner(Id, rho)
        current_E = real(inner(H, rho) / tr_rho)
        current_O = real(inner(O, rho) / tr_rho)

        # Branch termination condition depending on search direction
        if (dbeta > 0 && current_E <= exact_E) || (dbeta < 0 && current_E >= exact_E)
            break
        end
    end
    
    return beta, current_O
end

function disentangle_mps_to_product(psi_init::MPS)
    psi = copy(psi_init)
    N = length(psi)
    sites = siteinds(psi)
    
    # Array to store applied gates (for later use as a circuit)
    gates = ITensor[]

    for i in 1:(N-1)
        # 1. Move orthogonality center to the left side (i) of the current pair
        orthogonalize!(psi, i)
        
        s_left = sites[i]
        s_right = sites[i+1]
        
        # 2. Extract local wavefunction phi
        if i == 1
            # Left edge: Link is on the right side (l_right) only
            l_right = commonind(psi[i+1], psi[i+2])
            phi = psi[i] * psi[i+1]
            
            # Combiner for environment indices (links) (in this case, just keep it)
            C_links = combiner(l_right; tags="links_comb")
            
        elseif i == N-1
            # Right edge: Link is on the left side (l_left) only
            l_left = commonind(psi[i-1], psi[i])
            phi = psi[i] * psi[i+1]
            
            C_links = combiner(l_left; tags="links_comb")
            
        else
            # Bulk (middle): Links are on both sides
            l_left = commonind(psi[i-1], psi[i])
            l_right = commonind(psi[i+1], psi[i+2])
            phi = psi[i] * psi[i+1]
            
            C_links = combiner(l_left, l_right; tags="links_comb")
        end

        # 3. Calculate Source basis (current state)
        # Combine physical indices
        C_phys = combiner(s_left, s_right; tags="phys_comb")
        
        # Tensor for matrix conversion: (Phys) x (Links)
        M_tensor = phi * C_phys * C_links
        
        # Basis extraction by SVD
        # uniqueinds(..., combinedind(C_links)) puts physical side on rows, link side on cols
        U_active_tensor, S, V = svd(M_tensor, uniqueinds(M_tensor, combinedind(C_links)))
        
        # Convert to matrix form
        mat_src_active = Matrix(U_active_tensor, commonind(U_active_tensor, C_phys), commonind(U_active_tensor, S))
        
        # Calculate complement space to make a full unitary matrix
        mat_src_perp = nullspace(mat_src_active')
        mat_src_full = hcat(mat_src_active, mat_src_perp)

        # 4. Define Target basis (goal state)
        # Goal: Make the left site (i) be Up (Index=1)
        # ITensors basis order: [UpUp, UpDn, DnUp, DnDn]
        # Left side is Up for the first two: |Up,Up>, |Up,Dn>
        
        # Target basis vectors (4D)
        v1 = [1.0, 0, 0, 0] # |Up, Up>
        v2 = [0, 0, 1.0, 0] # |Up, Dn> (*Adjust here to change mapping destination)
        v3 = [0, 1.0, 0, 0] # |Dn, Up>
        v4 = [0, 0, 0, 1.0] # |Dn, Dn>
        
        mat_tgt_full = hcat(v1, v2, v3, v4)

        # 5. Gate construction G = Tgt * Src'
        mat_G = mat_tgt_full * mat_src_full'
        
        # Convert back to ITensor format
        c = combinedind(C_phys)
        G_c = itensor(mat_G, c', c)
        G = dag(C_phys') * G_c * C_phys
        
        push!(gates, G)
        
        # 6. Apply gate to MPS and update
        # apply(G, psi) may shift orthogonality center, manual update recommended but
        # here we use apply for simplicity and orthogonalize! in next loop
        psi = apply(G, psi)
        normalize!(psi)
        
    end
    
    return psi, gates
end

function apply_circuit(psi_0::MPS, gates::Vector{ITensor})
    psi = copy(psi_0)
    for G in gates
        psi = apply(G, psi)
        normalize!(psi)
    end
    return psi
end

function get_construction_gates(disentangling_gates::Vector{ITensor})
    construction_gates = ITensor[]
    
    # 1. Take out in reverse order
    for G in reverse(disentangling_gates)
        
        # 2. Create inverse unitary G† (Hermitian conjugate)
        # dag(G) takes complex conjugate, swapprime swaps input(0) and output(1) legs
        G_dag = swapprime(dag(G), 0 => 1)
        
        push!(construction_gates, G_dag)
    end
    
    return construction_gates
end

function itensor_to_matrix(T::ITensor, rows, cols)
    T_perm = permute(T, (rows..., cols...))
    return reshape(Array(T_perm, (rows..., cols...)), prod(dim.(rows)), prod(dim.(cols)))
end

function matrix_to_itensor(M::Matrix, rows, cols)
    T_array = reshape(M, (dim.(rows)..., dim.(cols)...))
    return itensor(T_array, (rows..., cols...))
end

function optimization_gates(psi_chimax::MPS, construction_gates::Vector{ITensor}, r::Float64)

    sites = siteinds(psi_chimax)
    N = length(sites)
    psi_up = MPS(sites, n->"Up")
    M_gates = length(construction_gates)
    new_construction_gates = copy(construction_gates)

    for m in 1:M_gates
        # Identify target sites
        U_target = new_construction_gates[m]
        s_gate = commoninds(U_target, sites)
        target_sites_indices = [findfirst(x->hasind(psi_chimax[x], s), 1:N) for s in s_gate]
        sort!(target_sites_indices)

        # (A) |psi_in> (Bra side): U_1 ... U_{m-1} |0>
        psi_bra = copy(psi_up)
        for j in 1:m-1
            psi_bra = apply(new_construction_gates[j], psi_bra)
            normalize!(psi_bra)
        end

        # (B) |psi_out> (Ket side): U_{m+1}^dag ... U_M^dag |psi_target>
        # [Fix] Apply inverse unitary (dag) when peeling from target
        psi_ket = copy(psi_chimax)
        for i in M_gates:-1:m+1
            G = new_construction_gates[i]
            # U -> U^dag (Swap input and output and take complex conjugate)
            G_dag = swapprime(dag(G), 0 => 1) 
            psi_ket = apply(G_dag, psi_ket)
            normalize!(psi_ket)
        end

        # (C) Construct MPO F_m_all = |psi_out><psi_in|
        # outer(A, B) creates |A><B|
        F_m_all = outer(psi_ket', psi_bra)

        # (D) Extract local environment F_m (Trace out unnecessary sites)
        L = ITensor(1.0)
        for i in 1:(target_sites_indices[1]-1)
            s = sites[i]
            L *= F_m_all[i] * delta(s, s')
        end

        R = ITensor(1.0)
        for i in N:-1:(target_sites_indices[end]+1)
            s = sites[i]
            R *= F_m_all[i] * delta(s, s')
        end

        Center = ITensor(1.0)
        for i in target_sites_indices
            Center *= F_m_all[i]
        end

        F_m = L * Center * R

        # --- 3. Gate Update (Algorithm 2) ---

        # Indices for matrix conversion
        row_inds = prime(s_gate) # s'
        col_inds = s_gate        # s

        M_F = itensor_to_matrix(F_m, row_inds, col_inds)
        M_U_old = itensor_to_matrix(U_target, row_inds, col_inds)

        # SVD: F_m = U S V^dag -> U_opt = U V^dag
        F_svd = svd(M_F)
        M_U_opt = F_svd.U * F_svd.Vt

        # Update rule: U_new = U_old * (U_old^dag * U_opt)^r
        M_W = M_U_old' * M_U_opt
        vals, vecs = eigen(M_W)
        vals_r = exp.(r .* log.(Complex.(vals)))
        M_W_r = vecs * Diagonal(vals_r) * inv(vecs)

        M_U_new = M_U_old * M_W_r
        usvd = svd(M_U_new)
        M_U_new = usvd.U * usvd.Vt
        U_new = matrix_to_itensor(M_U_new, row_inds, col_inds)

        new_construction_gates[m] = U_new
    end
    return new_construction_gates
end

function Iter_D_i_O_all(psi_chimax::MPS, layers::Int, r_init::Float64, optimization_sweeps::Int)
    
    sites = siteinds(psi_chimax)
    psi_up = MPS(sites, n->"Up")

    all_disentangling_gates = ITensor[]
    all_construction_gates = ITensor[]

    @showprogress for layer in 1:layers
        println("\n=== Layer $layer / $layers ===")
        
        # 1. Create residual state
        psi = copy(psi_chimax)
        psi = apply_circuit(psi, all_disentangling_gates)
        
        # 2. Approximate (truncate)
        psi_chi2 = copy(psi)
        truncate!(psi_chi2; maxdim=2)

        # 3. Calculate gates for new layer
        _, layer_gates = disentangle_mps_to_product(psi_chi2)
        append!(all_disentangling_gates, layer_gates)

        # 4. Convert to construction gates
        all_construction_gates = get_construction_gates(all_disentangling_gates)

        # --- Safe Optimization Loop (Best-Keep Strategy) ---
        
        # Calculate current best score
        psi_approx = copy(psi_up)
        psi_approx = apply_circuit(psi_approx, all_construction_gates)
        best_overlap = abs(inner(psi_approx, psi_chimax))
        best_gates = copy(all_construction_gates)
        
        println("  Initial Overlap: $best_overlap")

        current_r = r_init # Reset r for each layer, or decrease it

        for sweep in 1:optimization_sweeps
            # Trial update
            trial_gates = optimization_gates(psi_chimax, best_gates, current_r)
            
            # Evaluation
            psi_check = copy(psi_up)
            psi_check = apply_circuit(psi_check, trial_gates)
            current_overlap = abs(inner(psi_check, psi_chimax))

            if current_overlap > best_overlap
                # Adopt if improved
                # println("  Sweep $sweep: Improved ($best_overlap -> $current_overlap)")
                best_overlap = current_overlap
                best_gates = trial_gates
            else
                # Reject if worsened, decrease r to be cautious
                # println("  Sweep $sweep: Worsened ($current_overlap). Rejecting and reducing r.")
                current_r *= 0.5
                if current_r < 1e-3
                    break 
                end
            end
        end
        
        println("Final Overlap for Layer $layer: $best_overlap")
        
        # Adopt best gate set
        all_construction_gates = best_gates
        
        # 5. Return to disentangling form for next layer
        all_disentangling_gates = get_construction_gates(all_construction_gates)
    end

    return all_construction_gates
end

function main()

    L = 40
    chi = 10
    dt = 0.2
    rep = 150
    ttotal = dt * rep
    plotrep = 400
    maxdim_obs = 2048
    maxdim_state = 128
    trunc = 1e-7
    Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045/2, 0.0, 0.8090/2
    beta = 0.1
    lambda = 72.0/L^2
    observable = "Magz"
    parts = 30
    is = [20]
    layers = 5
    r = 0.6
    optimization_sweeps = 50
    # --- End of parameter settings ---

    # --- Reconstruct filename ---
    local cache_file
    cache_file = "$(observable)_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(parts).jld2"

    # --- Load Data ---
    local s, ts, O_Us
    if isfile(cache_file)
        println("File found: $cache_file")
        data = load(cache_file)
        s = data["s"]
        ts = data["ts"]
        O_Us = data["O_Us"]
        println("Data loading completed.")
    else
        println("File not found: $cache_file")
        return # quit if file does not exist
    end

    local O
    if observable == "Szc"
        c = div(L, 2) + 1 # center site
        os_Szc = OpSum()
        os_Szc += "Sz", c
        O = MPO(os_Szc,s) # Sz of site c
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

    plot_ts = (0:(plotrep - 1)) .* dt
    O_expected_exact = zeros(Float64, length(is), plotrep)
    EEs_exact = zeros(Float64, length(is), plotrep)
    O_expected_approx = zeros(Float64, length(is), plotrep)
    EEs_approx = zeros(Float64, length(is), plotrep)

    # ★Fix 4: Use enumerate to distinguish array index k and value i from is
    for (k, i) in enumerate(is)
        tau = ts[i+1]
        O_tau = O_Us[i+1]

        # H_dmrg = O_tau + H_penalty
        nsweeps = 50
        # maxdim = 1:10
        maxdim = fill(chi, nsweeps)
        # noise = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-12, 1e-13, 1e-13, 0.0]
        noise = [1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7,
                 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10, 1e-11, 1e-11, 1e-12, 1e-12, 1e-13, 1e-13, 1e-14, 1e-14, 0.0]

        println("Setting parameters: L=$L, lambda=$lambda, tau=$tau")

        Random.seed!(3001)
        psi0 = random_mps(ComplexF64, s; linkdims=2)
        if observable == "Magy"
            psi0 = MPS(ComplexF64, s, n -> "Y-")
        end
        if observable == "Magz"
            psi0 = MPS(ComplexF64, s, n -> "Dn")
        end
        for j in 1:i*div(rep, parts)
            psi0 = apply(gate_dag, psi0; maxdim=maxdim_state, cutoff=trunc)
            normalize!(psi0)
        end

        eigs, psi_exact = dmrg([O_tau, H_penalty], psi0; nsweeps, maxdim, noise, cutoff=trunc, outputlevel=0)

        psi_evolved = psi_exact
        for j in 1:plotrep
            O_expected_exact[k, j] = real(inner(psi_evolved', O, psi_evolved))
            EEs_exact[k, j] = entanglement_entropy(psi_evolved, div(L, 2))
            psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
            normalize!(psi_evolved)
        end

        psi_up = MPS(ComplexF64, s, n -> "Up")
        construction_gates = Iter_D_i_O_all(psi_exact, layers, r, optimization_sweeps)
        psi_approx = psi_up
        psi_approx = apply_circuit(psi_approx, construction_gates)
        println("Overlap between exact and approx: ", abs(inner(psi_exact, psi_approx)))

        beta_exact, current_O_exact = calc_beta_and_obs(L, s, psi_exact, O, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
        beta_approx, current_O_approx = calc_beta_and_obs(L, s, psi_approx, O, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
        println("Estimated beta (exact): ", beta_exact, ", O: ", current_O_exact)
        println("Estimated beta (approx): ", beta_approx, ", O: ", current_O_approx)

        psi_evolved = psi_approx
        for j in 1:plotrep
            # ★Fix 4: Use k instead of i for index
            O_expected_approx[k, j] = real(inner(psi_evolved', O, psi_evolved))
            EEs_approx[k, j] = entanglement_entropy(psi_evolved, div(L, 2))
            psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
            normalize!(psi_evolved)
        end
    end
    
    results_filename = "Burst_plot_QC_$(observable)_L$(L)_lambda$(lambda).jld2"
    println("Saving results to $results_filename...")
    save(results_filename, "plot_ts", plot_ts, "O_expected_exact", O_expected_exact, "EEs_exact", EEs_exact, "O_expected_approx", O_expected_approx, "EEs_approx", EEs_approx)
    println("Finished saving.")
end

main()
