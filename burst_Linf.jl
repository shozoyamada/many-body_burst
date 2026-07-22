using LinearAlgebra
using Printf
using ProgressMeter
using JLD2

# :z or :y
const CALC_MODE = :z

# Physical and numerical parameters
const d = 2
const sx = ComplexF64[0 1; 1 0] / 2
const sy = ComplexF64[0 -im; im 0] / 2
const sz = ComplexF64[1 0; 0 -1] / 2
const id2 = Matrix{ComplexF64}(I, d, d)

const MAXDIM_STATE = 128
const TARGET_CHI = 10
const DT = 0.2

# TEBD truncation settings.
# CUTOFF is a relative discarded-weight target. If MAXDIM_STATE is the limiting
# factor, the actual discarded weight can be larger and is reported.
const TEBD_CUTOFF = 1e-12
const THERMAL_CUTOFF = 1e-12
const SCHMIDT_RTOL = 1e-13
const SCHMIDT_ATOL = 1e-15

# Number of identity-gate sweeps after hard Schmidt truncation to TARGET_CHI.
const RECANON_SWEEPS = 4

println("Calculation mode: $CALC_MODE")

# =============================================================================
# Utility functions
# =============================================================================

function assert_finite(name::AbstractString, arrays...)
    for A in arrays
        all(isfinite, A) || error("Non-finite value detected in $name")
    end
    return nothing
end

function time_step_count(total_time::Real, dt::Real)
    nsteps = round(Int, total_time / dt)
    isapprox(nsteps * dt, total_time; atol=1e-12, rtol=1e-12) ||
        error("total_time=$total_time is not an integer multiple of dt=$dt")
    return nsteps
end

function choose_svd_rank(
    s::AbstractVector{<:Real};
    chi_max::Int,
    cutoff::Real,
    schmidt_rtol::Real,
    schmidt_atol::Real,
)
    isempty(s) && error("SVD returned no singular values")
    chi_max >= 1 || error("chi_max must be positive")

    weights = abs2.(s)
    total_weight = sum(weights)
    isfinite(total_weight) && total_weight > 0 ||
        error("Invalid singular-value norm: $total_weight")

    smax = maximum(abs, s)
    floor_value = max(schmidt_atol, schmidt_rtol * smax)
    numerical_rank = findlast(x -> abs(x) > floor_value, s)
    numerical_rank = isnothing(numerical_rank) ? 1 : numerical_rank
    max_keep = min(chi_max, numerical_rank)

    # Smallest rank whose total discarded weight meets cutoff, provided that
    # the bond-dimension cap permits it. Otherwise keep max_keep.
    rank = max_keep
    kept_weight = 0.0
    for k in 1:max_keep
        kept_weight += weights[k]
        discarded = max(total_weight - kept_weight, 0.0) / total_weight
        if discarded <= cutoff
            rank = k
            break
        end
    end

    discarded_weight = max(total_weight - sum(@view weights[1:rank]), 0.0) /
                       total_weight
    return rank, discarded_weight
end

function checked_inverse_schmidt(
    lambda::AbstractVector{<:Real};
    schmidt_rtol::Real,
    schmidt_atol::Real,
)
    isempty(lambda) && error("Empty Schmidt vector")
    scale = maximum(abs, lambda)
    threshold = max(schmidt_atol, schmidt_rtol * scale)
    minimum(abs, lambda) > threshold || error(
        "Retained Schmidt value below inversion threshold. " *
        "min=$(minimum(abs, lambda)), threshold=$threshold"
    )
    return 1.0 ./ lambda
end

# Vidal-form canonical residuals for a one-site tensor Gamma with left and right
# Schmidt vectors. Both should be close to zero for a canonical iMPS.
function canonical_residuals(Gamma, lambda_left, lambda_right)
    dloc, chi_left, chi_right = size(Gamma)
    length(lambda_left) == chi_left || error("Left bond mismatch")
    length(lambda_right) == chi_right || error("Right bond mismatch")

    # Sum_{s,b} Gamma[s,a,b] lambda_right[b]^2 Gamma*[s,a',b] = delta[a,a']
    X = Gamma .* reshape(lambda_right, (1, 1, chi_right))
    Xmat = reshape(permutedims(X, (2, 1, 3)), chi_left, dloc * chi_right)
    left_gram = Xmat * Xmat'
    left_res = norm(left_gram - Matrix{ComplexF64}(I, chi_left, chi_left)) /
               sqrt(chi_left)

    # Sum_{s,a} lambda_left[a]^2 Gamma*[s,a,b] Gamma[s,a,b'] = delta[b,b']
    Y = Gamma .* reshape(lambda_left, (1, chi_left, 1))
    Ymat = reshape(Y, dloc * chi_left, chi_right)
    right_gram = Ymat' * Ymat
    right_res = norm(right_gram - Matrix{ComplexF64}(I, chi_right, chi_right)) /
                sqrt(chi_right)

    return left_res, right_res
end

function unitcell_canonical_residual(GammaA, lambdaA, GammaB, lambdaB)
    # A has left bond lambdaB and right bond lambdaA; B is the reverse.
    rA = canonical_residuals(GammaA, lambdaB, lambdaA)
    rB = canonical_residuals(GammaB, lambdaA, lambdaB)
    return maximum((rA..., rB...))
end

# =============================================================================
# Initial states: exact rank-one product states
# =============================================================================

function initialize_all_z_down(dloc::Int)
    dloc == 2 || error("This initializer assumes a spin-1/2 local space")
    Gamma = zeros(ComplexF64, dloc, 1, 1)
    Gamma[2, 1, 1] = 1.0
    lambda = [1.0]
    return Gamma, lambda
end

function initialize_all_y_down(dloc::Int)
    dloc == 2 || error("This initializer assumes a spin-1/2 local space")
    Gamma = zeros(ComplexF64, dloc, 1, 1)
    # |y-> = (|0> - i|1>) / sqrt(2)
    Gamma[1, 1, 1] = 1 / sqrt(2)
    Gamma[2, 1, 1] = -im / sqrt(2)
    lambda = [1.0]
    return Gamma, lambda
end

# =============================================================================
# Stable two-site iTEBD update
# =============================================================================

function apply_gate_and_truncate(
    Gamma_L,
    lambda_center,
    Gamma_R,
    lambda_right,
    lambda_left,
    gate,
    chi_max::Int;
    cutoff::Real,
    schmidt_rtol::Real=SCHMIDT_RTOL,
    schmidt_atol::Real=SCHMIDT_ATOL,
)
    dphys, chi_left, chi_center = size(Gamma_L)
    dphys_R, chi_center_R, chi_right = size(Gamma_R)

    dphys == dphys_R || error("Physical dimensions do not match")
    chi_center == chi_center_R || error("Center bond dimensions do not match")
    length(lambda_left) == chi_left || error("lambda_left dimension mismatch")
    length(lambda_center) == chi_center || error("lambda_center dimension mismatch")
    length(lambda_right) == chi_right || error("lambda_right dimension mismatch")

    gate_mat = reshape(gate, dphys * dphys, dphys * dphys)

    # Build theta[sL,a,sR,c] = lambda_left[a] Gamma_L[sL,a,b]
    #                            lambda_center[b] Gamma_R[sR,b,c]
    #                            lambda_right[c].
    GLw = Gamma_L .* reshape(lambda_left, (1, chi_left, 1))
    GRw = Gamma_R .* reshape(lambda_right, (1, 1, chi_right))

    ML = reshape(GLw, dphys * chi_left, chi_center)
    ML .*= reshape(lambda_center, 1, chi_center)
    MR = reshape(permutedims(GRw, (2, 1, 3)), chi_center, dphys * chi_right)

    theta = reshape(ML * MR, dphys, chi_left, dphys, chi_right)

    theta_kron_order = permutedims(theta, (3, 1, 2, 4))
    theta_mat = reshape(theta_kron_order, dphys * dphys, chi_left * chi_right)
    evolved = gate_mat * theta_mat

    # evolved4 has indices (s_right, s_left, chi_left, chi_right).
    evolved4 = reshape(evolved, dphys, dphys, chi_left, chi_right)
    svd_matrix = reshape(permutedims(evolved4, (2, 3, 1, 4)),
                         dphys * chi_left, dphys * chi_right)

    F = svd(svd_matrix; full=false)
    rank, discarded_weight = choose_svd_rank(
        F.S;
        chi_max=chi_max,
        cutoff=cutoff,
        schmidt_rtol=schmidt_rtol,
        schmidt_atol=schmidt_atol,
    )

    Ukeep = @view F.U[:, 1:rank]
    Skeep = collect(@view F.S[1:rank])
    Vtkeep = @view F.Vt[1:rank, :]

    retained_norm = norm(Skeep)
    isfinite(retained_norm) && retained_norm > 0 ||
        error("Invalid retained SVD norm: $retained_norm")
    lambda_new = Skeep ./ retained_norm

    inv_left = checked_inverse_schmidt(
        lambda_left;
        schmidt_rtol=schmidt_rtol,
        schmidt_atol=schmidt_atol,
    )
    inv_right = checked_inverse_schmidt(
        lambda_right;
        schmidt_rtol=schmidt_rtol,
        schmidt_atol=schmidt_atol,
    )

    Gamma_L_new = reshape(Matrix(Ukeep), dphys, chi_left, rank)
    Gamma_L_new .*= reshape(inv_left, 1, chi_left, 1)

    Gamma_R_new = reshape(Matrix(Vtkeep), rank, dphys, chi_right)
    Gamma_R_new = permutedims(Gamma_R_new, (2, 1, 3))
    Gamma_R_new .*= reshape(inv_right, 1, 1, chi_right)

    assert_finite("two-site update", Gamma_L_new, lambda_new, Gamma_R_new)

    diagnostics = (
        discarded_weight=discarded_weight,
        kept_rank=rank,
        retained_norm=retained_norm,
        min_schmidt=minimum(lambda_new),
    )
    return Gamma_L_new, lambda_new, Gamma_R_new, diagnostics
end

function apply_second_order_step(
    GammaA,
    lambdaA,
    GammaB,
    lambdaB,
    gate_half,
    gate_full,
    chi_max::Int;
    cutoff::Real,
)
    GammaA, lambdaA, GammaB, d1 = apply_gate_and_truncate(
        GammaA, lambdaA, GammaB, lambdaB, lambdaB, gate_half, chi_max;
        cutoff=cutoff,
    )

    GammaB, lambdaB, GammaA, d2 = apply_gate_and_truncate(
        GammaB, lambdaB, GammaA, lambdaA, lambdaA, gate_full, chi_max;
        cutoff=cutoff,
    )

    GammaA, lambdaA, GammaB, d3 = apply_gate_and_truncate(
        GammaA, lambdaA, GammaB, lambdaB, lambdaB, gate_half, chi_max;
        cutoff=cutoff,
    )

    diagnostics = (
        discarded_sum=d1.discarded_weight + d2.discarded_weight + d3.discarded_weight,
        discarded_max=max(d1.discarded_weight, d2.discarded_weight, d3.discarded_weight),
        max_rank=max(d1.kept_rank, d2.kept_rank, d3.kept_rank),
        min_schmidt=min(d1.min_schmidt, d2.min_schmidt, d3.min_schmidt),
    )
    return GammaA, lambdaA, GammaB, lambdaB, diagnostics
end

# =============================================================================
# Compression and recanonicalization
# =============================================================================

function identity_gate(dloc::Int)
    return reshape(Matrix{ComplexF64}(I, dloc * dloc, dloc * dloc),
                   dloc, dloc, dloc, dloc)
end

function truncate_and_recanonicalize(
    GammaA,
    lambdaA,
    GammaB,
    lambdaB,
    chi_target::Int;
    nsweeps::Int=RECANON_SWEEPS,
)
    rankA = min(chi_target, length(lambdaA))
    rankB = min(chi_target, length(lambdaB))

    normA2 = sum(abs2, lambdaA)
    normB2 = sum(abs2, lambdaB)
    errA = rankA < length(lambdaA) ? sum(abs2, @view lambdaA[rankA+1:end]) / normA2 : 0.0
    errB = rankB < length(lambdaB) ? sum(abs2, @view lambdaB[rankB+1:end]) / normB2 : 0.0

    lambdaA_new = collect(@view lambdaA[1:rankA])
    lambdaB_new = collect(@view lambdaB[1:rankB])
    lambdaA_new ./= norm(lambdaA_new)
    lambdaB_new ./= norm(lambdaB_new)

    # GammaA: left bond B, right bond A. GammaB: left A, right B.
    GammaA_new = copy(@view GammaA[:, 1:rankB, 1:rankA])
    GammaB_new = copy(@view GammaB[:, 1:rankA, 1:rankB])

    gate_id = identity_gate(size(GammaA_new, 1))
    recanon_discard = 0.0
    for _ in 1:nsweeps
        # One AB update and one BA update are sufficient for an identity sweep.
        GammaA_new, lambdaA_new, GammaB_new, dAB = apply_gate_and_truncate(
            GammaA_new, lambdaA_new, GammaB_new,
            lambdaB_new, lambdaB_new, gate_id, chi_target;
            cutoff=0.0,
        )
        GammaB_new, lambdaB_new, GammaA_new, dBA = apply_gate_and_truncate(
            GammaB_new, lambdaB_new, GammaA_new,
            lambdaA_new, lambdaA_new, gate_id, chi_target;
            cutoff=0.0,
        )
        recanon_discard += dAB.discarded_weight + dBA.discarded_weight
    end

    canonical_residual = unitcell_canonical_residual(
        GammaA_new, lambdaA_new, GammaB_new, lambdaB_new
    )

    diagnostics = (
        truncation_error=0.5 * (errA + errB),
        recanonicalization_discarded=recanon_discard,
        canonical_residual=canonical_residual,
        rankA=length(lambdaA_new),
        rankB=length(lambdaB_new),
    )
    return GammaA_new, lambdaA_new, GammaB_new, lambdaB_new, diagnostics
end

# =============================================================================
# Measurements
# =============================================================================

function measure_1site(Gamma, lambda_left, lambda_right, op)
    dloc, chi_left, chi_right = size(Gamma)
    psi = Gamma .* reshape(lambda_left, (1, chi_left, 1)) .*
          reshape(lambda_right, (1, 1, chi_right))
    psi_mat = reshape(psi, dloc, chi_left * chi_right)
    norm_sq = real(dot(psi_mat, psi_mat))
    isfinite(norm_sq) && norm_sq > 0 || error("Invalid one-site norm")
    return real(dot(psi_mat, op * psi_mat) / norm_sq)
end

function measure_2site(GammaL, lambdaC, GammaR, lambdaR, lambdaL, op)
    dloc, chi_left, chi_center = size(GammaL)
    dloc_R, chi_center_R, chi_right = size(GammaR)
    dloc == dloc_R || error("Physical dimension mismatch")
    chi_center == chi_center_R || error("Center dimension mismatch")

    GLw = GammaL .* reshape(lambdaL, (1, chi_left, 1))
    GRw = GammaR .* reshape(lambdaR, (1, 1, chi_right))
    ML = reshape(GLw, dloc * chi_left, chi_center)
    ML .*= reshape(lambdaC, 1, chi_center)
    MR = reshape(permutedims(GRw, (2, 1, 3)), chi_center, dloc * chi_right)

    theta = reshape(ML * MR, dloc, chi_left, dloc, chi_right)
    # Standard kron ordering has the right-site index varying fastest.
    theta_phys = permutedims(theta, (3, 1, 2, 4))
    theta_mat = reshape(theta_phys, dloc * dloc, chi_left * chi_right)

    norm_sq = real(dot(theta_mat, theta_mat))
    isfinite(norm_sq) && norm_sq > 0 || error("Invalid two-site norm")
    return real(dot(theta_mat, op * theta_mat) / norm_sq)
end

function measure_energy(GammaA, lambdaA, GammaB, lambdaB, Hbond)
    EAB = measure_2site(GammaA, lambdaA, GammaB, lambdaB, lambdaB, Hbond)
    EBA = measure_2site(GammaB, lambdaB, GammaA, lambdaA, lambdaA, Hbond)
    return 0.5 * (EAB + EBA)
end

# =============================================================================
# Thermal purification and inverse-temperature matching
# =============================================================================

function construct_thermal_gate(Hbond, dphys::Int, dbeta_state::Real)
    Uphys = exp(-dbeta_state * Hbond)
    D = dphys * dphys
    Gmat = zeros(ComplexF64, D * D, D * D)

    # Combined local basis: k = (physical-1)*dphys + ancilla.
    # Two-site matrix basis: row = (k_left-1)*D + k_right.
    for p1 in 1:dphys, p2 in 1:dphys, l1 in 1:dphys, l2 in 1:dphys
        row_phys = (p1 - 1) * dphys + p2
        col_phys = (l1 - 1) * dphys + l2
        u = Uphys[row_phys, col_phys]
        for a1 in 1:dphys, a2 in 1:dphys
            k1 = (p1 - 1) * dphys + a1
            k2 = (p2 - 1) * dphys + a2
            m1 = (l1 - 1) * dphys + a1
            m2 = (l2 - 1) * dphys + a2
            row = (k1 - 1) * D + k2
            col = (m1 - 1) * D + m2
            Gmat[row, col] = u
        end
    end
    return reshape(Gmat, D, D, D, D)
end

function expand_two_site_operator_for_purification(op, dphys::Int)
    D = dphys * dphys
    out = zeros(ComplexF64, D * D, D * D)
    for p1 in 1:dphys, p2 in 1:dphys, l1 in 1:dphys, l2 in 1:dphys
        row_phys = (p1 - 1) * dphys + p2
        col_phys = (l1 - 1) * dphys + l2
        val = op[row_phys, col_phys]
        for a1 in 1:dphys, a2 in 1:dphys
            k1 = (p1 - 1) * dphys + a1
            k2 = (p2 - 1) * dphys + a2
            m1 = (l1 - 1) * dphys + a1
            m2 = (l2 - 1) * dphys + a2
            row = (k1 - 1) * D + k2
            col = (m1 - 1) * D + m2
            out[row, col] = val
        end
    end
    return out
end

function expand_one_site_operator_for_purification(op, dphys::Int)
    D = dphys * dphys
    out = zeros(ComplexF64, D, D)
    for p1 in 1:dphys, p2 in 1:dphys, a in 1:dphys
        k1 = (p1 - 1) * dphys + a
        k2 = (p2 - 1) * dphys + a
        out[k1, k2] = op[p1, p2]
    end
    return out
end

function initialize_infinite_temperature_purification(dphys::Int)
    D = dphys * dphys
    Gamma = zeros(ComplexF64, D, 1, 1)
    for i in 1:dphys
        k = (i - 1) * dphys + i
        Gamma[k, 1, 1] = 1 / sqrt(dphys)
    end
    return Gamma, [1.0]
end

function thermal_observables(GammaA, lambdaA, GammaB, lambdaB, mz_exp, my_exp)
    mzA = measure_1site(GammaA, lambdaB, lambdaA, mz_exp)
    mzB = measure_1site(GammaB, lambdaA, lambdaB, mz_exp)
    myA = measure_1site(GammaA, lambdaB, lambdaA, my_exp)
    myB = measure_1site(GammaB, lambdaA, lambdaB, my_exp)
    return 0.5 * (mzA + mzB), 0.5 * (myA + myB)
end

function find_beta_and_magnetizations(
    GammaA,
    lambdaA,
    GammaB,
    lambdaB,
    Hbond,
    dphys::Int,
    chi_max::Int;
    beta_max::Real=1.0,
    dbeta_phys_abs::Real=1e-4,
    energy_tol::Real=1e-10,
)
    E_target = measure_energy(GammaA, lambdaA, GammaB, lambdaB, Hbond)

    GammaAt, lambdaAt = initialize_infinite_temperature_purification(dphys)
    GammaBt, lambdaBt = initialize_infinite_temperature_purification(dphys)

    Hexp = expand_two_site_operator_for_purification(Hbond, dphys)
    mz_exp = expand_one_site_operator_for_purification(sz, dphys)
    my_exp = expand_one_site_operator_for_purification(sy, dphys)

    E_prev = measure_energy(GammaAt, lambdaAt, GammaBt, lambdaBt, Hexp)
    mz_prev, my_prev = thermal_observables(
        GammaAt, lambdaAt, GammaBt, lambdaBt, mz_exp, my_exp
    )
    beta_prev = 0.0

    if abs(E_prev - E_target) <= energy_tol
        return (
            beta=0.0,
            mz=mz_prev,
            my=my_prev,
            energy_residual=E_prev - E_target,
            thermal_discarded=0.0,
            bracket_width=0.0,
            energy_bracket_span=0.0,
        )
    end

    step_sign = E_target < E_prev ? 1.0 : -1.0

    # Purification state is exp(-beta H / 2)|I>, hence half the physical beta step.
    dbeta_state = step_sign * dbeta_phys_abs / 2
    gate_full = construct_thermal_gate(Hbond, dphys, dbeta_state)
    gate_half = construct_thermal_gate(Hbond, dphys, dbeta_state / 2)

    max_steps = floor(Int, beta_max / dbeta_phys_abs + 1e-12)
    thermal_discarded = 0.0

    for _ in 1:max_steps
        # Keep references to the previous state. The update routines allocate
        # new tensors, so these arrays remain valid without an explicit copy.
        GammaAt_prev, lambdaAt_prev = GammaAt, lambdaAt
        GammaBt_prev, lambdaBt_prev = GammaBt, lambdaBt
        discarded_before_step = thermal_discarded

        GammaAt, lambdaAt, GammaBt, lambdaBt, diag = apply_second_order_step(
            GammaAt, lambdaAt, GammaBt, lambdaBt,
            gate_half, gate_full, chi_max;
            cutoff=THERMAL_CUTOFF,
        )
        discarded_after_full_step = discarded_before_step + diag.discarded_sum

        beta_now = beta_prev + step_sign * dbeta_phys_abs
        E_now = measure_energy(GammaAt, lambdaAt, GammaBt, lambdaBt, Hexp)

        crossed = step_sign > 0 ? E_now <= E_target : E_now >= E_target
        if crossed || abs(E_now - E_target) <= energy_tol
            denom = E_now - E_prev
            alpha = abs(denom) <= eps(Float64) ? 1.0 : (E_target - E_prev) / denom
            alpha = clamp(alpha, 0.0, 1.0)
            beta_final = beta_prev + alpha * (beta_now - beta_prev)

            if alpha <= 10 * eps(Float64)
                GammaAf, lambdaAf = GammaAt_prev, lambdaAt_prev
                GammaBf, lambdaBf = GammaBt_prev, lambdaBt_prev
                discarded_final = discarded_before_step
            elseif alpha >= 1 - 10 * eps(Float64)
                GammaAf, lambdaAf = GammaAt, lambdaAt
                GammaBf, lambdaBf = GammaBt, lambdaBt
                discarded_final = discarded_after_full_step
            else
                # Re-evolve from the previous endpoint by the fractional beta
                # interval. This makes the returned observables state-based.
                dbeta_phys_fraction = alpha * step_sign * dbeta_phys_abs
                dbeta_state_fraction = dbeta_phys_fraction / 2
                gate_full_fraction = construct_thermal_gate(
                    Hbond, dphys, dbeta_state_fraction
                )
                gate_half_fraction = construct_thermal_gate(
                    Hbond, dphys, dbeta_state_fraction / 2
                )
                GammaAf, lambdaAf, GammaBf, lambdaBf, fracdiag =
                    apply_second_order_step(
                        GammaAt_prev, lambdaAt_prev, GammaBt_prev, lambdaBt_prev,
                        gate_half_fraction, gate_full_fraction, chi_max;
                        cutoff=THERMAL_CUTOFF,
                    )
                discarded_final = discarded_before_step + fracdiag.discarded_sum
            end

            E_final = measure_energy(GammaAf, lambdaAf, GammaBf, lambdaBf, Hexp)
            mz_final, my_final = thermal_observables(
                GammaAf, lambdaAf, GammaBf, lambdaBf, mz_exp, my_exp
            )

            return (
                beta=beta_final,
                mz=mz_final,
                my=my_final,
                energy_residual=E_final - E_target,
                thermal_discarded=discarded_final,
                bracket_width=abs(beta_now - beta_prev),
                energy_bracket_span=abs(E_now - E_prev),
            )
        end

        thermal_discarded = discarded_after_full_step
        beta_prev = beta_now
        E_prev = E_now
        mz_prev, my_prev = thermal_observables(
            GammaAt, lambdaAt, GammaBt, lambdaBt, mz_exp, my_exp
        )
    end

    error(
        "Could not bracket target energy E=$E_target within |beta| <= $beta_max. " *
        "Last thermal energy was $E_prev at beta=$beta_prev."
    )
end

# =============================================================================
# Hamiltonian and gates
# =============================================================================

# The paper uses H = sum SzSz + (1/2) sum(hx Sx + hz Sz).
# In an infinite two-site bond decomposition each onsite term is split equally
# between its two neighboring bonds.
Jx, Jy, Jz = 0.0, 0.0, 1.0
hx_paper, hy_paper, hz_paper = 0.9045, 0.0, 0.8090
hx_coeff, hy_coeff, hz_coeff = hx_paper / 2, hy_paper / 2, hz_paper / 2

H_bond = Jx * kron(sx, sx) + Jy * kron(sy, sy) + Jz * kron(sz, sz) +
         0.5 * hx_coeff * (kron(sx, id2) + kron(id2, sx)) +
         0.5 * hy_coeff * (kron(sy, id2) + kron(id2, sy)) +
         0.5 * hz_coeff * (kron(sz, id2) + kron(id2, sz))

Gate_full = reshape(exp(-im * H_bond * DT), d, d, d, d)
Gate_half = reshape(exp(-im * H_bond * (DT / 2)), d, d, d, d)
Gate_full_dag = reshape(exp(im * H_bond * DT), d, d, d, d)
Gate_half_dag = reshape(exp(im * H_bond * (DT / 2)), d, d, d, d)

if CALC_MODE == :z
    init_func = initialize_all_z_down
    observable_op = sz
    extract_eq = result -> result.mz
    observable_name = "Magz"
elseif CALC_MODE == :y
    init_func = initialize_all_y_down
    observable_op = sy
    extract_eq = result -> result.my
    observable_name = "Magy"
else
    error("Invalid CALC_MODE=$CALC_MODE. Choose :z or :y.")
end

# =============================================================================
# Main calculation
# =============================================================================

println("Starting stable thermodynamic-limit burst simulation...")

taus = collect(0.0:1.0:30.0)
exp_vals = Float64[]
eq_vals = Float64[]
beta_vals = Float64[]
truncation_errors = Float64[]
recanon_residuals = Float64[]
backward_discarded = Float64[]
forward_discarded = Float64[]
thermal_discarded = Float64[]
thermal_energy_residuals = Float64[]
thermal_energy_bracket_spans = Float64[]
final_canonical_residuals = Float64[]

@showprogress for tau in taus
    GammaA, lambdaA = init_func(d)
    GammaB, lambdaB = init_func(d)

    nsteps = time_step_count(tau, DT)

    # 1. Time-reversed evolution at the large working bond dimension.
    back_disc = 0.0
    for _ in 1:nsteps
        GammaA, lambdaA, GammaB, lambdaB, diag = apply_second_order_step(
            GammaA, lambdaA, GammaB, lambdaB,
            Gate_half_dag, Gate_full_dag, MAXDIM_STATE;
            cutoff=TEBD_CUTOFF,
        )
        back_disc += diag.discarded_sum
    end
    push!(backward_discarded, back_disc)

    # 2. Truncate both Schmidt bonds to TARGET_CHI and recanonicalize.
    GammaA, lambdaA, GammaB, lambdaB, comp = truncate_and_recanonicalize(
        GammaA, lambdaA, GammaB, lambdaB, TARGET_CHI;
        nsweeps=RECANON_SWEEPS,
    )
    push!(truncation_errors, comp.truncation_error)
    push!(recanon_residuals, comp.canonical_residual)

    # 3. Match the energy with a thermal state.
    thermal = find_beta_and_magnetizations(
        GammaA, lambdaA, GammaB, lambdaB,
        H_bond, d, MAXDIM_STATE;
        beta_max=1.0,
        dbeta_phys_abs=1e-4,
        energy_tol=1e-10,
    )
    push!(beta_vals, thermal.beta)
    push!(eq_vals, extract_eq(thermal))
    push!(thermal_discarded, thermal.thermal_discarded)
    push!(thermal_energy_residuals, thermal.energy_residual)
    push!(thermal_energy_bracket_spans, thermal.energy_bracket_span)

    # 4. Forward evolution from the compressed state.
    fwd_disc = 0.0
    for _ in 1:nsteps
        GammaA, lambdaA, GammaB, lambdaB, diag = apply_second_order_step(
            GammaA, lambdaA, GammaB, lambdaB,
            Gate_half, Gate_full, MAXDIM_STATE;
            cutoff=TEBD_CUTOFF,
        )
        fwd_disc += diag.discarded_sum
    end
    push!(forward_discarded, fwd_disc)

    expA = measure_1site(GammaA, lambdaB, lambdaA, observable_op)
    expB = measure_1site(GammaB, lambdaA, lambdaB, observable_op)
    push!(exp_vals, 0.5 * (expA + expB))
    push!(final_canonical_residuals,
          unitcell_canonical_residual(GammaA, lambdaA, GammaB, lambdaB))
end

burst_vals = eq_vals .- exp_vals

@printf("max compression error             = %.6e\n", maximum(truncation_errors))
@printf("max post-compression canon. resid. = %.6e\n", maximum(recanon_residuals))
@printf("max final canonical residual       = %.6e\n", maximum(final_canonical_residuals))
@printf("max |thermal energy residual|      = %.6e\n", maximum(abs, thermal_energy_residuals))
@printf("max thermal energy bracket span   = %.6e\n", maximum(thermal_energy_bracket_spans))

results_filename = "inftauvsburst_$(observable_name)_T$(maximum(taus))_chi$(TARGET_CHI).jld2"
println("Saving results to $results_filename")

parameters = Dict(
    "calc_mode" => string(CALC_MODE),
    "maxdim_state" => MAXDIM_STATE,
    "target_chi" => TARGET_CHI,
    "dt" => DT,
    "tebd_cutoff" => TEBD_CUTOFF,
    "thermal_cutoff" => THERMAL_CUTOFF,
    "schmidt_rtol" => SCHMIDT_RTOL,
    "schmidt_atol" => SCHMIDT_ATOL,
    "recanonicalization_sweeps" => RECANON_SWEEPS,
    "Jz" => Jz,
    "hx_paper" => hx_paper,
    "hz_paper" => hz_paper,
)

save(
    results_filename,
    "taus", taus,
    "burst_vals", burst_vals,
    "exp_vals", exp_vals,
    "eq_vals", eq_vals,
    "beta_vals", beta_vals,
    "truncation_errors", truncation_errors,
    "recanonicalization_residuals", recanon_residuals,
    "backward_discarded", backward_discarded,
    "forward_discarded", forward_discarded,
    "thermal_discarded", thermal_discarded,
    "thermal_energy_residuals", thermal_energy_residuals,
    "thermal_energy_bracket_spans", thermal_energy_bracket_spans,
    "final_canonical_residuals", final_canonical_residuals,
    "parameters", parameters,
)

println("Simulation completed.")
