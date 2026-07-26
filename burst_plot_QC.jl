using MKL
using LinearAlgebra, Statistics, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter
BLAS.set_num_threads(Threads.nthreads())

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

function calc_beta_and_obs(N, s, psi, O, Jx, Jy, Jz, hx, hy, hz; dbeta_abs=0.0001, max_steps=10000, maxdim=2048, cutoff=1e-14)

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

# -----------------------------------------------------------------------------
# Fast MPS -> shallow-circuit decomposition / optimization
#
# Main speedup relative to the original implementation:
#   * Never constructs the full MPO |psi_out><psi_in| for every gate.
#   * Reuses left/right contraction environments while sweeping consecutively.
#   * Applies a whole circuit with one `apply` call and normalizes only once.
#
# The sweep order follows the construction-gate convention used below: each
# layer is stored from the rightmost bond to the leftmost bond.
# -----------------------------------------------------------------------------

"""Return the inverse of a gate while restoring input/output prime levels."""
inverse_gate(G::ITensor) = swapprime(dag(G), 0 => 1)

"""
Apply a two-site gate directly to neighboring tensors of an MPS.

`psi` must have its orthogonality center at `b`. After the SVD, the center is
moved to `b + 1`, so repeated calls with increasing `b` form an efficient sweep.
"""
function apply_two_site_gate!(
    psi::MPS,
    G::ITensor,
    b::Int;
    maxdim::Int=typemax(Int),
    cutoff::Float64=0.0,
)
    1 <= b < length(psi) || throw(BoundsError(psi, b))

    left_inds = uniqueinds(psi[b], psi[b + 1])
    theta = noprime(G * psi[b] * psi[b + 1])
    U, S, V = svd(theta, left_inds; maxdim=maxdim, cutoff=cutoff)
    psi[b] = U
    psi[b + 1] = S * V
    return psi
end

function disentangle_mps_to_product(
    psi_init::MPS;
    maxdim::Int=max(2, maxlinkdim(psi_init)),
    cutoff::Float64=1e-14,
)
    psi = copy(psi_init)
    N = length(psi)
    sites = siteinds(psi)

    gates = ITensor[]

    for i in 1:(N - 1)
        orthogonalize!(psi, i)

        s_left = sites[i]
        s_right = sites[i + 1]

        if i == 1
            l_right = commonind(psi[i + 1], psi[i + 2])
            phi = psi[i] * psi[i + 1]
            C_links = combiner(l_right; tags="links_comb")
        elseif i == N - 1
            l_left = commonind(psi[i - 1], psi[i])
            phi = psi[i] * psi[i + 1]
            C_links = combiner(l_left; tags="links_comb")
        else
            l_left = commonind(psi[i - 1], psi[i])
            l_right = commonind(psi[i + 1], psi[i + 2])
            phi = psi[i] * psi[i + 1]
            C_links = combiner(l_left, l_right; tags="links_comb")
        end

        C_phys = combiner(s_left, s_right; tags="phys_comb")
        M_tensor = phi * C_phys * C_links

        U_active_tensor, S, _ = svd(
            M_tensor,
            uniqueinds(M_tensor, combinedind(C_links)),
        )

        mat_src_active = Matrix(
            U_active_tensor,
            commonind(U_active_tensor, C_phys),
            commonind(U_active_tensor, S),
        )

        # Complete the isometry to a 4x4 unitary.
        mat_src_perp = nullspace(mat_src_active')
        mat_src_full = hcat(mat_src_active, mat_src_perp)

        # ITensor's two-spin basis ordering used by the original code.
        v1 = ComplexF64[1, 0, 0, 0]
        v2 = ComplexF64[0, 0, 1, 0]
        v3 = ComplexF64[0, 1, 0, 0]
        v4 = ComplexF64[0, 0, 0, 1]
        mat_tgt_full = hcat(v1, v2, v3, v4)

        mat_G = mat_tgt_full * mat_src_full'

        c = combinedind(C_phys)
        G_c = itensor(mat_G, c', c)
        G = dag(C_phys') * G_c * C_phys
        push!(gates, G)

        # The original version called generic `apply` and normalized after every
        # gate. A local update is cheaper and preserves the sweep gauge.
        apply_two_site_gate!(psi, G, i; maxdim=maxdim, cutoff=cutoff)
    end

    normalize!(psi)
    return psi, gates
end

"""Apply a gate list to an MPS with a single contraction/truncation pass."""
function apply_circuit(
    psi_0::MPS,
    gates::AbstractVector{<:ITensor};
    maxdim::Int=typemax(Int),
    cutoff::Float64=0.0,
)
    psi = copy(psi_0)
    isempty(gates) || (psi = apply(gates, psi; maxdim=maxdim, cutoff=cutoff))
    normalize!(psi)
    return psi
end

function get_construction_gates(disentangling_gates::AbstractVector{<:ITensor})
    construction_gates = ITensor[]
    sizehint!(construction_gates, length(disentangling_gates))
    for G in reverse(disentangling_gates)
        push!(construction_gates, inverse_gate(G))
    end
    return construction_gates
end

function itensor_to_matrix(T::ITensor, rows, cols)
    T_perm = permute(T, (rows..., cols...))
    return reshape(
        Array(T_perm, (rows..., cols...)),
        prod(dim.(rows)),
        prod(dim.(cols)),
    )
end

function matrix_to_itensor(M::AbstractMatrix, rows, cols)
    T_array = reshape(M, (dim.(rows)..., dim.(cols)...))
    return itensor(T_array, (rows..., cols...))
end

"""Nearest unitary to a square matrix in Frobenius norm."""
function polar_unitary(M::AbstractMatrix)
    F = svd(M)
    return F.U * F.Vt
end

"""Equation (6) of Rudolph et al.: U <- U (U' Uopt)^r."""
function update_gate_from_environment(
    U_old::ITensor,
    F_env::ITensor,
    s_left::Index,
    s_right::Index,
    r::Float64,
)
    row_inds = (s_left', s_right')
    col_inds = (s_left, s_right)

    M_F = itensor_to_matrix(F_env, row_inds, col_inds)
    M_U_old = itensor_to_matrix(U_old, row_inds, col_inds)

    M_U_opt = polar_unitary(M_F)

    # Project to U(4) before taking a fractional power. This keeps numerical
    # drift from pushing eigenvalues away from the unit circle.
    M_W = polar_unitary(M_U_old' * M_U_opt)
    E = eigen(M_W)
    phases_r = cis.(r .* angle.(E.values))
    M_W_r = E.vectors * Diagonal(phases_r) * inv(E.vectors)

    M_U_new = polar_unitary(M_U_old * M_W_r)
    return matrix_to_itensor(M_U_new, row_inds, col_inds)
end

"""Find the left bond index (1 ... N-1) on which a two-site gate acts."""
function gate_bond(G::ITensor, sites)
    s_gate = collect(commoninds(G, sites))
    length(s_gate) == 2 || error("Expected a two-site gate, got $(length(s_gate)) site indices")

    positions = sort([
        something(findfirst(==(s), sites), 0)
        for s in s_gate
    ])
    positions[1] > 0 || error("Gate contains a site index not present in the MPS")
    positions[2] == positions[1] + 1 || error("Only nearest-neighbor gates are supported")
    return positions[1]
end

"""
Validate the layer ordering produced by `get_construction_gates`.

Each construction layer must be stored as bonds N-1, N-2, ..., 1. The cached
optimizer traverses that block in reverse, i.e. left-to-right in physical space.
"""
function validate_construction_order(bonds::Vector{Int}, N::Int)
    gates_per_layer = N - 1
    length(bonds) % gates_per_layer == 0 || error(
        "Gate count $(length(bonds)) is not a multiple of N-1=$(gates_per_layer)",
    )

    expected = collect(gates_per_layer:-1:1)
    for first_gate in 1:gates_per_layer:length(bonds)
        block = bonds[first_gate:(first_gate + gates_per_layer - 1)]
        block == expected || error(
            "Unexpected gate order in a layer. Expected bonds $expected, got $block",
        )
    end
    return nothing
end

"""
Build reusable contractions of sites b+2 ... N for a left-to-right gate sweep.

`ket` and `bra` have the same physical indices but independent MPS link indices.
`R[b]` is precisely the part to the right of a gate acting on (b,b+1).
"""
function build_right_overlap_environments(ket::MPS, bra::MPS)
    N = length(ket)
    R = Vector{ITensor}(undef, N - 1)
    env = ITensor(1.0)

    for b in (N - 1):-1:1
        R[b] = env
        site = b + 1
        env = (ket[site] * dag(bra[site])) * env
    end
    return R
end

"""Two-site environment tensor corresponding to Eq. (5) of the paper."""
function local_gate_environment(
    ket::MPS,
    bra::MPS,
    b::Int,
    Lenv::ITensor,
    Renv::ITensor,
)
    center = prime(ket[b], "Site") * dag(bra[b])
    center *= prime(ket[b + 1], "Site") * dag(bra[b + 1])
    return Lenv * center * Renv
end

"""
Perform one cached forward optimization sweep over all current circuit layers.

The old implementation rebuilt both partial circuit states and a full outer MPO
for every gate. Here each layer gets one right-environment build, and both MPS
frontiers are advanced locally as gates are visited consecutively.
"""
function optimization_sweep_cached(
    psi_target::MPS,
    construction_gates::AbstractVector{<:ITensor},
    psi_circuit::MPS,
    gate_bonds::Vector{Int},
    r::Float64;
    maxdim::Int=typemax(Int),
    cutoff::Float64=0.0,
)
    sites = siteinds(psi_target)
    N = length(sites)
    gates_per_layer = N - 1
    M = length(construction_gates)
    num_layers = M ÷ gates_per_layer

    new_gates = copy(construction_gates)

    # `psi_prefix` starts as U_M...U_1|0>. During the reverse circuit sweep,
    # old gates are peeled off. `psi_target_side` starts as |target>; updated
    # adjoints are applied to it. Their local overlap is Eq. (5).
    psi_prefix = copy(psi_circuit)
    psi_target_side = copy(psi_target)

    for layer in num_layers:-1:1
        block_first = (layer - 1) * gates_per_layer + 1
        block_last = layer * gates_per_layer

        orthogonalize!(psi_prefix, 1)
        orthogonalize!(psi_target_side, 1)
        right_envs = build_right_overlap_environments(psi_target_side, psi_prefix)
        left_env = ITensor(1.0)

        # Stored block is N-1,...,1; reverse traversal visits bonds 1,...,N-1.
        for m in block_last:-1:block_first
            b = gate_bonds[m]

            # Remove the old gate from the circuit output.
            apply_two_site_gate!(
                psi_prefix,
                inverse_gate(new_gates[m]),
                b;
                maxdim=maxdim,
                cutoff=cutoff,
            )

            F_env = local_gate_environment(
                psi_target_side,
                psi_prefix,
                b,
                left_env,
                right_envs[b],
            )

            U_new = update_gate_from_environment(
                new_gates[m],
                F_env,
                sites[b],
                sites[b + 1],
                r,
            )
            new_gates[m] = U_new

            # Advance the target side with the freshly updated inverse gate.
            apply_two_site_gate!(
                psi_target_side,
                inverse_gate(U_new),
                b;
                maxdim=maxdim,
                cutoff=cutoff,
            )

            # Site b is finalized for this layer and can be cached into L.
            left_env *= psi_target_side[b] * dag(psi_prefix[b])
        end
    end

    # Reconstruct once per sweep, rather than once for every gate.
    psi_up = MPS(ComplexF64, sites, n -> "Up")
    trial_state = apply_circuit(
        psi_up,
        new_gates;
        maxdim=maxdim,
        cutoff=cutoff,
    )
    trial_overlap = abs(inner(trial_state, psi_target))
    return new_gates, trial_state, trial_overlap
end

# Backward-compatible one-sweep interface matching the original function name.
function optimization_gates(
    psi_chimax::MPS,
    construction_gates::AbstractVector{<:ITensor},
    r::Float64;
    maxdim::Int=maxlinkdim(psi_chimax),
    cutoff::Float64=1e-12,
)
    sites = siteinds(psi_chimax)
    psi_up = MPS(ComplexF64, sites, n -> "Up")
    psi_circuit = apply_circuit(
        psi_up,
        construction_gates;
        maxdim=maxdim,
        cutoff=cutoff,
    )
    bonds = [gate_bond(G, sites) for G in construction_gates]
    validate_construction_order(bonds, length(sites))
    new_gates, _, _ = optimization_sweep_cached(
        psi_chimax,
        construction_gates,
        psi_circuit,
        bonds,
        r;
        maxdim=maxdim,
        cutoff=cutoff,
    )
    return new_gates
end

function Iter_D_i_O_all(
    psi_chimax::MPS,
    layers::Int,
    r_init::Float64,
    optimization_sweeps::Int;
    maxdim::Int=maxlinkdim(psi_chimax),
    cutoff::Float64=1e-12,
    min_r::Float64=1e-3,
    overlap_tol::Float64=1e-12,
)
    sites = siteinds(psi_chimax)
    N = length(sites)
    psi_up = MPS(ComplexF64, sites, n -> "Up")

    all_disentangling_gates = ITensor[]
    all_construction_gates = ITensor[]

    @showprogress for layer in 1:layers
        println("\n=== Layer $layer / $layers ===")

        # Algorithm 1: analytically initialize one new layer from the residual.
        psi_residual = apply_circuit(
            psi_chimax,
            all_disentangling_gates;
            maxdim=maxdim,
            cutoff=cutoff,
        )
        psi_chi2 = copy(psi_residual)
        truncate!(psi_chi2; maxdim=2, cutoff=cutoff)

        _, layer_gates = disentangle_mps_to_product(
            psi_chi2;
            maxdim=2,
            cutoff=cutoff,
        )
        append!(all_disentangling_gates, layer_gates)
        all_construction_gates = get_construction_gates(all_disentangling_gates)

        gate_bonds = [gate_bond(G, sites) for G in all_construction_gates]
        validate_construction_order(gate_bonds, N)

        # Algorithm 2: optimize all existing layers, retaining the best sweep.
        best_state = apply_circuit(
            psi_up,
            all_construction_gates;
            maxdim=maxdim,
            cutoff=cutoff,
        )
        best_overlap = abs(inner(best_state, psi_chimax))
        best_gates = copy(all_construction_gates)
        println("  Initial overlap: $best_overlap")

        current_r = r_init
        for sweep in 1:optimization_sweeps
            trial_gates, trial_state, trial_overlap = optimization_sweep_cached(
                psi_chimax,
                best_gates,
                best_state,
                gate_bonds,
                current_r;
                maxdim=maxdim,
                cutoff=cutoff,
            )

            if trial_overlap > best_overlap + overlap_tol
                best_overlap = trial_overlap
                best_gates = trial_gates
                best_state = trial_state
            else
                current_r *= 0.5
                current_r < min_r && break
            end
        end

        println("Final overlap for layer $layer: $best_overlap")
        all_construction_gates = best_gates
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
    Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045 / 2, 0.0, 0.8090 / 2
    beta = 0.1
    penalty_coeff = 72.0
    lambda = penalty_coeff / L^2
    observable = "Magz"
    num_parts = 30
    is = [20]
    layers = 5
    r = 0.6
    optimization_sweeps = 50
    num_trials = 3
    # --- End of parameter settings ---

    # --- Reconstruct filename (same convention as the other UOU scripts) ---
    cache_file = "submit_UOU/$(observable)_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"

    # --- Load data ---
    local s, ts, O_Us
    if isfile(cache_file)
        println("File found: $cache_file")
        data = load(cache_file)
        s = data["s"]
        ts = data["ts"]
        O_Us = data["O_Us"]
        println("Finished loading data.")
    else
        println("File not found: $cache_file")
        return
    end

    local O
    if observable == "Szc"
        c = div(L, 2) + 1
        os_Szc = OpSum()
        os_Szc += "Sz", c
        O = MPO(os_Szc, s)
    elseif observable == "Magy"
        O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 1 / L, 0.0)
    elseif observable == "Magz"
        O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / L)
    else
        error("Unsupported observable: $observable")
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

    # Metadata for the selected exact trial and the circuit approximation.
    selected_trials = zeros(Int, length(is))
    selected_burst_values = zeros(Float64, length(is))
    selected_eq_values = zeros(Float64, length(is))
    selected_dyn_values = zeros(Float64, length(is))
    beta_exact_values = zeros(Float64, length(is))
    beta_approx_values = zeros(Float64, length(is))
    eq_approx_values = zeros(Float64, length(is))
    exact_approx_overlaps = zeros(Float64, length(is))

    for (k, i) in enumerate(is)
        tau_index = i + 1
        tau = ts[tau_index]
        O_tau = O_Us[tau_index]
        evolution_steps = i * div(rep, num_parts)

        nsweeps = 50
        maxdim = fill(chi, nsweeps)
        noise_schedule = [
            1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6,
            1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10,
            1e-11, 1e-11, 1e-12, 1e-12, 1e-13, 1e-13, 1e-14, 1e-14, 0.0,
        ]

        println("Setting parameters: L=$L, lambda=$lambda, tau=$tau")

        # Run DMRG three times and keep the state giving the largest burst.
        current_max_burst = -Inf
        best_eq_val = 0.0
        best_dyn_val = 0.0
        best_beta = 0.0
        best_trial = 0
        best_psi_exact = nothing

        for trial in 1:num_trials
            local psi0

            # Match the initialization policy of the other two scripts:
            # random product MPS for the first trials, and the backward-evolved
            # observable-aligned product state for the final trial.
            if trial != num_trials
                Random.seed!(1000 + L * 100 + tau_index * 10 + trial)
                psi0 = random_mps(ComplexF64, s; linkdims=1)
            else
                if observable == "Magy"
                    psi0 = MPS(ComplexF64, s, n -> "Y-")
                else
                    psi0 = MPS(ComplexF64, s, n -> "Dn")
                end

                for _ in 1:evolution_steps
                    psi0 = apply(gate_dag, psi0; maxdim=maxdim_state, cutoff=trunc)
                    normalize!(psi0)
                end
            end

            _, psi_trial = dmrg(
                [O_tau, H_penalty],
                psi0;
                nsweeps=nsweeps,
                maxdim=maxdim,
                noise=noise_schedule,
                outputlevel=0,
            )

            psi_at_tau = copy(psi_trial)
            for _ in 1:evolution_steps
                psi_at_tau = apply(gate, psi_at_tau; maxdim=maxdim_state, cutoff=trunc)
                normalize!(psi_at_tau)
            end

            beta_trial, current_O = calc_beta_and_obs(
                L,
                s,
                psi_trial,
                O,
                Jx,
                Jy,
                Jz,
                hx,
                hy,
                hz;
                maxdim=maxdim_obs,
            )

            dyn_val = -real(inner(psi_at_tau', O, psi_at_tau))
            burst_val = current_O + dyn_val

            println(
                "  trial=$trial, beta=$beta_trial, eq=$current_O, " *
                "dyn=$dyn_val, burst=$burst_val",
            )

            if burst_val > current_max_burst
                current_max_burst = burst_val
                best_eq_val = current_O
                best_dyn_val = dyn_val
                best_beta = beta_trial
                best_trial = trial
                best_psi_exact = copy(psi_trial)
            end
        end

        psi_exact = best_psi_exact::MPS
        selected_trials[k] = best_trial
        selected_burst_values[k] = current_max_burst
        selected_eq_values[k] = best_eq_val
        selected_dyn_values[k] = best_dyn_val
        beta_exact_values[k] = best_beta

        println(
            "Selected trial $best_trial / $num_trials: " *
            "burst=$current_max_burst (eq=$best_eq_val, dyn=$best_dyn_val)",
        )

        psi_evolved = copy(psi_exact)
        for j in 1:plotrep
            O_expected_exact[k, j] = real(inner(psi_evolved', O, psi_evolved))
            EEs_exact[k, j] = entanglement_entropy(psi_evolved, div(L, 2))
            psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
            normalize!(psi_evolved)
        end

        psi_up = MPS(ComplexF64, s, n -> "Up")
        construction_gates = Iter_D_i_O_all(
            psi_exact,
            layers,
            r,
            optimization_sweeps;
            maxdim=maxdim_state,
            cutoff=trunc,
        )
        psi_approx = apply_circuit(
            psi_up,
            construction_gates;
            maxdim=maxdim_state,
            cutoff=trunc,
        )

        overlap_exact_approx = abs(inner(psi_exact, psi_approx))
        exact_approx_overlaps[k] = overlap_exact_approx
        println("Overlap between exact and approx: $overlap_exact_approx")

        beta_approx, current_O_approx = calc_beta_and_obs(
            L,
            s,
            psi_approx,
            O,
            Jx,
            Jy,
            Jz,
            hx,
            hy,
            hz;
            maxdim=maxdim_obs,
        )
        beta_approx_values[k] = beta_approx
        eq_approx_values[k] = current_O_approx

        println("Estimated beta (exact): $best_beta, O: $best_eq_val")
        println("Estimated beta (approx): $beta_approx, O: $current_O_approx")

        psi_evolved = copy(psi_approx)
        for j in 1:plotrep
            O_expected_approx[k, j] = real(inner(psi_evolved', O, psi_evolved))
            EEs_approx[k, j] = entanglement_entropy(psi_evolved, div(L, 2))
            psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
            normalize!(psi_evolved)
        end
    end

    results_filename = "Burst_plot_QC_$(observable)_L$(L)_lambda$(penalty_coeff)_chi$(chi)_trials$(num_trials).jld2"
    println("Saving results to $results_filename...")
    save(
        results_filename,
        "plot_ts", plot_ts,
        "is", is,
        "selected_trials", selected_trials,
        "selected_burst_values", selected_burst_values,
        "selected_eq_values", selected_eq_values,
        "selected_dyn_values", selected_dyn_values,
        "beta_exact_values", beta_exact_values,
        "beta_approx_values", beta_approx_values,
        "eq_approx_values", eq_approx_values,
        "exact_approx_overlaps", exact_approx_overlaps,
        "O_expected_exact", O_expected_exact,
        "EEs_exact", EEs_exact,
        "O_expected_approx", O_expected_approx,
        "EEs_approx", EEs_approx,
    )
    println("Finished saving.")
    flush(stdout)
end

main()
