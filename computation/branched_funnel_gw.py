"""
Branched Funnel Goldberger-Wise Stabilization
==============================================
Source Protocol computation: Do junction matching conditions
on a branched I × S² geometry produce lepton mass gaps at ln(4π)?

The key idea: Z₈ holonomy selects different S² modes for each branch.
Different mode spectra → different Casimir energies → different IR brane positions.

Setup:
  Trunk: z ∈ [0, ε]  (short common UV region)
  Branch j: z ∈ [ε, L_j]  (three separate funnels)
  Junction at z = ε: continuity + flux conservation

  Z₈ charge j selects S² modes with m ≡ j (mod 8).
  Branch 0 (tau):  m=0 modes → ℓ ≥ 0  (most modes → deepest potential → shortest)
  Branch 1 (muon): m=1 modes → ℓ ≥ 1  (fewer modes → shallower → longer)
  Branch 2 (electron): m=2 modes → ℓ ≥ 2  (fewest modes → shallowest → longest)

  Lighter particles = longer funnels. The missing low-ℓ modes make the
  potential shallower, pushing the IR brane further out.

  Question: does the gap equal ln(4π)?
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar, brentq, minimize
from math import pi, log, exp, sqrt

# ============================================================
# Physical constants and SP parameters
# ============================================================
k = 1.0        # warp parameter (units: everything in units of k)
nu_0 = 2.1     # ν = √(4 + m²/k²), GW bulk mass parameter (ν > 2 for stability)
delta = nu_0 - 2  # δ = ν - 2 = 0.1

v_UV = 1.0     # UV brane VEV
v_IR = 0.1     # IR brane VEV (same for all branches)
eps = 0.01     # trunk length (short)

# SP geometric values
m_phi_over_k = 1 / (2 * sqrt(3))  # G4: m_φ/k = 1/(2√3) ≈ 0.2887
ln4pi = log(4 * pi)               # ≈ 2.531

# Observed
m_e = 0.51100  # MeV
m_mu = 105.658
m_tau = 1776.86
kL_e_obs = log(1.221e19 * 1e3 / m_e)  # kL_e from M_Pl to m_e
gap_mu_tau_obs = log(m_tau / m_mu)      # ≈ 2.822
gap_e_mu_obs = log(m_mu / m_e)         # ≈ 5.332

print("=" * 70)
print("BRANCHED FUNNEL GW STABILIZATION")
print("=" * 70)
print(f"\nSP parameters: ν = {nu_0}, δ = {delta}, m_φ/k = {m_phi_over_k:.4f}")
print(f"ln(4π) = {ln4pi:.6f}")
print(f"\nObserved gaps:")
print(f"  gap(μ→τ) = ln(m_τ/m_μ) = {gap_mu_tau_obs:.4f}")
print(f"  gap(e→μ) = ln(m_μ/m_e) = {gap_e_mu_obs:.4f}")
print(f"  ratio = {gap_e_mu_obs/gap_mu_tau_obs:.4f}")
print(f"\nC1 predictions:")
print(f"  gap(μ→τ) = ln(4π) + m_φ/k = {ln4pi + m_phi_over_k:.4f}")
print(f"  gap(e→μ) = 2ln(4π) + m_φ/k = {2*ln4pi + m_phi_over_k:.4f}")

# ============================================================
# PART 1: GW equation on a single segment
# ============================================================

def gw_coeffs(z_start, z_end, phi_start, phi_end, nu):
    """Solve GW equation φ'' - 4kφ' + m²φ = 0 on [z_s, z_e].
    Solution: φ(z) = e^{2kz}[A e^{νkz} + B e^{-νkz}]
    Returns (A, B).
    """
    fp_s = exp((2 + nu) * z_start)
    fm_s = exp((2 - nu) * z_start)
    fp_e = exp((2 + nu) * z_end)
    fm_e = exp((2 - nu) * z_end)

    det = fp_s * fm_e - fm_s * fp_e
    if abs(det) < 1e-300:
        return None, None
    A = (phi_start * fm_e - phi_end * fm_s) / det
    B = (phi_end * fp_s - phi_start * fp_e) / det
    return A, B


def gw_deriv(z, A, B, nu):
    """φ'(z) = k e^{2kz}[(2+ν)A e^{νkz} + (2-ν)B e^{-νkz}]"""
    return exp(2 * z) * ((2 + nu) * A * exp(nu * z) + (2 - nu) * B * exp(-nu * z))


def segment_onshell(z_s, z_e, phi_s, phi_e, nu):
    """On-shell action for GW scalar on [z_s, z_e].
    V = [e^{-4kz} φ φ']_{z_s}^{z_e}
    """
    A, B = gw_coeffs(z_s, z_e, phi_s, phi_e, nu)
    if A is None:
        return 1e10
    dphi_s = gw_deriv(z_s, A, B, nu)
    dphi_e = gw_deriv(z_e, A, B, nu)
    return exp(-4 * z_e) * phi_e * dphi_e - exp(-4 * z_s) * phi_s * dphi_s


# ============================================================
# PART 2: Single funnel baseline
# ============================================================

print("\n" + "=" * 70)
print("PART 1: SINGLE FUNNEL GW POTENTIAL (BASELINE)")
print("=" * 70)

def V_single(L, nu=nu_0, v0=v_UV, v1=v_IR):
    """GW potential for single funnel of length L."""
    if L <= 0.01:
        return 1e10
    return segment_onshell(0, L, v0, v1, nu)

# Find minimum
from scipy.optimize import minimize_scalar
res = minimize_scalar(V_single, bounds=(10, 80), method='bounded')
L_min = res.x
V_min = res.fun
print(f"\nSingle funnel minimum: kL = {L_min:.4f}")
print(f"V(L_min) = {V_min:.6e}")

# Standard GW formula: kL = (1/δ) ln(v_UV/v_IR)
kL_gw = (1 / delta) * log(v_UV / v_IR)
print(f"GW formula kL = (1/δ)ln(v₀/v₁) = {kL_gw:.4f}")

# ============================================================
# PART 3: GW with angular momentum barrier from S² modes
# ============================================================

print("\n" + "=" * 70)
print("PART 2: S² MODES — ℓ-DEPENDENT GW POTENTIAL")
print("=" * 70)

def V_ell_numerical(L, ell, r0=1.0, nu=nu_0, v0=v_UV, v1=v_IR, Npts=500):
    """GW potential for mode ℓ on S², solved numerically.

    The equation with angular barrier:
    φ'' - 4kφ' + [m² - ℓ(ℓ+1)e^{2kz}/r₀²] φ = 0

    This is NOT solvable in closed form due to the z-dependent barrier.
    """
    if L <= 0.1:
        return 0.0

    z = np.linspace(0, L, Npts)

    # Effective mass squared including angular barrier
    m2_bulk = (nu**2 - 4)  # m²/k² = ν² - 4

    def ode(z, y):
        phi, dphi = y
        # φ'' = 4k φ' - m²_eff φ
        # m²_eff = m² - ℓ(ℓ+1)e^{2kz}/r₀²
        barrier = ell * (ell + 1) * np.exp(2 * z) / r0**2
        ddphi = 4 * dphi - (m2_bulk - barrier) * phi
        return np.array([dphi, ddphi])

    def bc(ya, yb):
        return np.array([ya[0] - v0, yb[0] - v1])

    # Initial guess: exponential interpolation
    z_mesh = np.linspace(0, L, 50)
    y_guess = np.zeros((2, 50))
    for i, zi in enumerate(z_mesh):
        t = zi / L
        y_guess[0, i] = v0 * (1 - t) + v1 * t
        y_guess[1, i] = (v1 - v0) / L

    try:
        sol = solve_bvp(ode, bc, z_mesh, y_guess, tol=1e-8, max_nodes=5000)
        if not sol.success:
            return 0.0

        # On-shell action from boundary terms
        phi_0, dphi_0 = sol.sol(0)
        phi_L, dphi_L = sol.sol(L)
        V = exp(-4 * L) * phi_L * dphi_L - phi_0 * dphi_0
        return V
    except:
        return 0.0


# Compute ℓ=0,1,2 potentials as function of L
print("\nComputing ℓ-dependent potentials...")
L_range = np.linspace(15, 60, 100)

V_l0 = np.array([V_ell_numerical(L, 0) for L in L_range])
V_l1 = np.array([V_ell_numerical(L, 1) for L in L_range])
V_l2 = np.array([V_ell_numerical(L, 2) for L in L_range])

# Find minima for each ℓ
for ell, V_arr, label in [(0, V_l0, 'ℓ=0'), (1, V_l1, 'ℓ=1'), (2, V_l2, 'ℓ=2')]:
    valid = V_arr < 1e9
    if valid.any():
        idx = np.argmin(V_arr[valid])
        L_vals = L_range[valid]
        print(f"  V_{label} minimum at kL ≈ {L_vals[idx]:.2f}, V = {V_arr[valid][idx]:.6e}")

# ============================================================
# PART 4: Branch-dependent mode spectrum from Z₈
# ============================================================

print("\n" + "=" * 70)
print("PART 3: Z₈ MODE SELECTION → BRANCH-DEPENDENT CASIMIR POTENTIAL")
print("=" * 70)

print("""
Z₈ acts on S² via rotation by π/4. Spherical harmonics Y_ℓm transform as:
  Y_ℓm → e^{imπ/4} Y_ℓm

Branch j (Z₈ charge j) selects modes with m ≡ j (mod 8):
  Branch 0 (tau):     m=0  → one mode per ℓ ≥ 0
  Branch 1 (muon):    m=1  → one mode per ℓ ≥ 1
  Branch 2 (electron): m=2  → one mode per ℓ ≥ 2

The Casimir energy difference between branches determines IR brane shifts.
""")

def count_modes(j, ell_max=20):
    """Count allowed S² modes for branch j up to ℓ_max.
    Allowed: m ≡ j (mod 8), with |m| ≤ ℓ
    """
    modes = []
    for ell in range(0, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m % 8 == j % 8 or (m - j) % 8 == 0:
                modes.append((ell, m))
    return modes

for j in range(3):
    modes = count_modes(j, 10)
    mode_str = ', '.join([f'({l},{m})' for l, m in modes[:8]])
    print(f"  Branch {j}: {len(modes)} modes up to ℓ=10: {mode_str}...")

# ============================================================
# PART 5: Total potential for each branch
# ============================================================

print("\n" + "=" * 70)
print("PART 4: BRANCH POTENTIALS AND IR BRANE POSITIONS")
print("=" * 70)

def V_branch(L, j, ell_max=6, r0=1.0):
    """Total GW potential for branch j, summing over allowed S² modes.

    Branch j includes modes (ℓ, m) with m ≡ j mod 8 and |m| ≤ ℓ.
    Each mode contributes V_ℓ(L) to the total potential.

    Key: the angular eigenvalue is ℓ(ℓ+1), same for all m at given ℓ.
    So modes with same ℓ but different m contribute identical V_ℓ(L).
    """
    V_total = 0
    mode_count = 0
    for ell in range(0, ell_max + 1):
        # Count allowed m values at this ℓ
        n_m = 0
        for m in range(-ell, ell + 1):
            if (m - j) % 8 == 0:
                n_m += 1
        if n_m > 0:
            V_ell = V_ell_numerical(L, ell, r0)
            V_total += n_m * V_ell
            mode_count += n_m
    return V_total

# Scan each branch
print(f"\nScanning branch potentials (ℓ_max=6, r₀=1.0)...")
print(f"This sums V_ℓ(L) weighted by the Z₈-allowed mode count at each ℓ.\n")

L_scan = np.linspace(20, 60, 80)
branch_names = ['tau (j=0)', 'muon (j=1)', 'electron (j=2)']
L_minima = []

for j in range(3):
    V_arr = np.array([V_branch(L, j) for L in L_scan])
    valid = np.isfinite(V_arr) & (V_arr < 1e9) & (V_arr != 0)
    if valid.any():
        idx = np.argmin(V_arr[valid])
        L_best = L_scan[valid][idx]
        V_best = V_arr[valid][idx]
        L_minima.append(L_best)
        print(f"  Branch {j} ({branch_names[j]}): kL_min ≈ {L_best:.2f}, V = {V_best:.4e}")
    else:
        print(f"  Branch {j} ({branch_names[j]}): no valid minimum found")
        L_minima.append(None)

# Compute gaps
if all(L is not None for L in L_minima):
    print(f"\n--- Mass gaps from branched funnel ---")
    gap_01 = L_minima[1] - L_minima[0] if L_minima[1] and L_minima[0] else None
    gap_12 = L_minima[2] - L_minima[1] if L_minima[2] and L_minima[1] else None
    gap_02 = L_minima[2] - L_minima[0] if L_minima[2] and L_minima[0] else None

    if gap_01 is not None:
        print(f"  kL(muon) - kL(tau)     = {gap_01:.4f}  (target: {gap_mu_tau_obs:.4f})")
        print(f"    / ln(4π) = {gap_01/ln4pi:.4f}")
    if gap_12 is not None:
        print(f"  kL(electron) - kL(muon) = {gap_12:.4f}  (target: {gap_e_mu_obs:.4f})")
        print(f"    / ln(4π) = {gap_12/ln4pi:.4f}")
    if gap_02 is not None:
        print(f"  kL(electron) - kL(tau)  = {gap_02:.4f}  (target: {gap_mu_tau_obs + gap_e_mu_obs:.4f})")
    if gap_01 and gap_12 and gap_01 != 0:
        print(f"  Ratio (e-μ)/(μ-τ) = {gap_12/gap_01:.4f}  (target: {gap_e_mu_obs/gap_mu_tau_obs:.4f})")

# ============================================================
# PART 6: Analytic estimate — Casimir shift from missing modes
# ============================================================

print("\n" + "=" * 70)
print("PART 5: ANALYTIC ESTIMATE — CASIMIR SHIFT FROM MISSING MODES")
print("=" * 70)

print("""
The shift in L from removing the ℓ=0 mode:

In the GW mechanism, the potential minimum is at:
  kL = (1/δ) ln(v₀/v₁) + corrections

The ℓ-dependent correction comes from the one-loop determinant:
  δkL_ℓ = contribution from mode ℓ to the effective potential

For the Laplacian on S² with eigenvalue ℓ(ℓ+1) and degeneracy (2ℓ+1):
The spectral zeta function:
  ζ_{S²}(s) = Σ_{ℓ=1}^∞ (2ℓ+1) / [ℓ(ℓ+1)]^s

The functional determinant:
  ln det(-Δ_{S²}) = -ζ'_{S²}(0)

For a unit S²: ζ'_{S²}(0) = 4ζ'_R(-1) - ln(2π)
  where ζ_R is the Riemann zeta function.
  ζ'_R(-1) = 1/12 - ln(A) where A = Glaisher-Kinkelin ≈ 1.282
""")

from scipy.special import zeta as riemann_zeta

# Glaisher-Kinkelin constant
A_GK = 1.2824271291  # Glaisher-Kinkelin constant
zeta_prime_neg1 = 1/12 - log(A_GK)
print(f"ζ'_R(-1) = {zeta_prime_neg1:.6f}")

# Functional determinant of -Δ on S²
# ln det(-Δ_{S²}) = -ζ'_{S²}(0)
# For S² of radius r: eigenvalues λ_ℓ = ℓ(ℓ+1)/r²
# ζ'_{S²}(0) is the relevant quantity

# The key quantity: contribution of each ℓ mode to the effective action
# For a scalar on warped S² × interval, the one-loop effective potential gets:
#   V_1loop(L) = -(1/2) Σ_ℓ (2ℓ+1) ln[det(-∂_z² + ℓ(ℓ+1)/R(z)²)]
# where R(z) = r₀ e^{-kz}

# The shift in kL from mode ℓ:
# At the GW minimum, V'(L) = 0. Adding the one-loop correction:
# V'(L) + V'_1loop(L) = 0
# δL = -V'_1loop(L*) / V''(L*)

# For the standard GW potential:
# V(L) ≈ C₁ e^{-(4+2δ)kL} + C₂ e^{-4kL}
# V'(L) = -(4+2δ)k C₁ e^{-(4+2δ)kL} - 4k C₂ e^{-4kL}
# V''(L) = (4+2δ)²k² C₁ e^{-(4+2δ)kL} + 16k² C₂ e^{-4kL}

# At minimum: V'(L*) = 0 → C₁ e^{-2δkL*} = -4C₂/(4+2δ)
# V''(L*) = 2δ(4+2δ)k² |C₁| e^{-(4+2δ)kL*}

# The shift from removing the ℓ=0 mode:
# This mode contributes a fraction of the total potential
# The fraction depends on the S² radius r₀

# KEY INSIGHT: The contribution of the ℓ-th mode to the GW potential
# involves the S² area through the angular eigenvalue ℓ(ℓ+1).
# The RATIO of contributions between successive ℓ modes should
# involve 4π = Area(S²) through the heat kernel.

# Heat kernel on S²:
# K(t) = Σ_ℓ (2ℓ+1) e^{-ℓ(ℓ+1)t} = 1/(4πt) [1 + t/3 + ...]  for small t
# The leading term 1/(4πt) → 1/t arises from the S² area = 4π

# The ratio of the ℓ=0 to ℓ=1 contribution to the heat kernel at t=1:
# K_0/K_1 = 1/3 * e^{2·1} = e²/3 ≈ 2.46...
# Not ln(4π). Let me think differently.

# The S² determinant enters as:
# For each branch, the stabilized position is:
# kL_j = kL_0 + (1/δ) × [change in ln det from missing modes]

# The change from removing mode (ℓ,m) from the S² sum:
# Δ(ln det) = ln(ℓ(ℓ+1)) for each removed mode

# Branch 0 (tau): all modes present → kL_0
# Branch 1 (muon): missing ℓ=0 mode → kL_0 + (1/δ) × ln(regularized)
# Branch 2 (electron): missing ℓ=0 and ℓ=1 → kL_0 + (1/δ) × ln(...)

# The ℓ=0 mode regularized eigenvalue: λ₀ = 0 → use zeta regularization
# The regularized product Π'(S²) = exp(-ζ'_{S²}(0))

# Actually, let me compute directly:
# The functional determinant of -Δ_{S²} (excluding zero mode):
# ln det'(-Δ_{S²}) = -Σ_{ℓ=1}^∞ (2ℓ+1) × d/ds[ℓ(ℓ+1)]^{-s}|_{s=0}
#                   = Σ_{ℓ=1}^∞ (2ℓ+1) × ln[ℓ(ℓ+1)]

# This diverges! Need zeta regularization.
# ζ_{S²}(s) = Σ_{ℓ=1}^∞ (2ℓ+1) / [ℓ(ℓ+1)]^s
# ln det'(-Δ_{S²}) = -ζ'_{S²}(0)

# Known result for unit S²:
# ζ'_{S²}(0) = 4ζ'_R(-1) - ln(2π) = 4(1/12 - ln A) - ln(2π)
zeta_prime_S2 = 4 * zeta_prime_neg1 - log(2 * pi)
log_det_S2 = -zeta_prime_S2
print(f"\nζ'_{{S²}}(0) = {zeta_prime_S2:.6f}")
print(f"ln det'(-Δ_{{S²}}) = {log_det_S2:.6f}")
print(f"det'(-Δ_{{S²}}) = {exp(log_det_S2):.6f}")
print(f"For comparison: 4π = {4*pi:.6f}")
print(f"                ln(4π) = {log(4*pi):.6f}")

# The one-loop shift in kL from the full S² determinant:
# δ(kL) = ½ ln(det'(-Δ_{S²})) ... but this is for the full determinant
# What about the shift from REMOVING one mode?

# If we remove the ℓ=0 mode (for branch 1 vs branch 0):
# The ℓ=0 mode on S² has eigenvalue 0, degeneracy 1
# Removing it from the regularized determinant:
# Δ ln det = ln(regularized ℓ=0 contribution)

# For the GW scalar on a segment of length L, with bulk equation
# -φ'' + 4kφ' + [ℓ(ℓ+1)e^{2kz}/r₀² - m²]φ = 0
# The one-loop determinant is:
# ln det(-∂²_z + V_ℓ(z)) where V_ℓ(z) = ℓ(ℓ+1)e^{2kz}/r₀² + ...

# The SHIFT in the GW minimum from the ℓ-th mode:
# The ℓ-th mode modifies the effective potential by:
# V_eff(L) = V_tree(L) + Σ_ℓ n_ℓ × V_1loop,ℓ(L)

# For the heat kernel approach:
# V_1loop,ℓ(L) = -(1/2) ∫_0^∞ dt/t × Tr[e^{-t(-∂² + V_ℓ)}]

# Let me try a different approach: direct perturbative shift.

print("\n--- Perturbative shift estimate ---")

# The ℓ-th mode's contribution to V_eff(L) is proportional to:
# ∫₀ᴸ dz e^{-4kz} × ℓ(ℓ+1) e^{2kz}/r₀² × φ²(z)
# = ℓ(ℓ+1)/r₀² × ∫₀ᴸ dz e^{-2kz} φ²(z)

# For the tree-level solution φ(z) ≈ v₀ e^{(2-ν)kz}:
# ∫₀ᴸ dz e^{-2kz} × v₀² e^{2(2-ν)kz} = v₀² ∫₀ᴸ dz e^{(2-2ν)kz}
# = v₀²/(2-2ν)k × [e^{(2-2ν)kL} - 1]
# For ν > 1: this is dominated by the z=0 end (UV)

# The shift in the minimum:
# δL ≈ -(dV_ℓ/dL) / (d²V_tree/dL²)

# For the GW potential V_tree(L):
# The dominant L-dependence comes from e^{-4kL} and e^{-(4+2δ)kL}
# d²V/dL² ∝ 2δ × 4k² × e^{-4kL} (at the minimum)

# The ℓ correction:
# dV_ℓ/dL ∝ ℓ(ℓ+1)/r₀² × e^{-(2+2ν)kL} × v₀²
# This involves the S² eigenvalue and the warp factor

# The ratio:
# δL ∝ ℓ(ℓ+1) / (r₀² × 2δ × 4k²) × e^{(4-2-2ν)kL} = ℓ(ℓ+1)/(8δr₀²) × e^{-2δkL}

# For small δ (δ ≈ 0.1) and large kL (kL ≈ 50):
# e^{-2δkL} = e^{-10} ≈ 4.5×10⁻⁵

# This is TINY. The angular barrier perturbation is exponentially suppressed!
# This means the ℓ-dependent shift is negligible for large kL.

L_test = 50
e_suppression = exp(-2 * delta * L_test)
print(f"e^{{-2δkL}} at kL=50: {e_suppression:.2e} — exponentially suppressed!")
print(f"This means the angular barrier perturbation to the GW minimum")
print(f"is negligible. The ℓ modes DO NOT shift the IR brane positions")
print(f"significantly in the standard perturbative framework.")

# ============================================================
# PART 7: Non-perturbative approach — the junction flux equation
# ============================================================

print("\n" + "=" * 70)
print("PART 6: JUNCTION FLUX EQUATION — NON-PERTURBATIVE")
print("=" * 70)

print("""
The perturbative shift fails because the angular barrier is exponentially
suppressed in the IR. But the JUNCTION is in the UV (z = ε ≈ 0), where
the S² is large and the barrier matters!

At the junction z = ε, the matching conditions for mode ℓ are:
  Continuity: φ_trunk^ℓ(ε) = φ_branch^ℓ(ε)  for each branch
  Flux conservation: (φ')_trunk^ℓ(ε) = Σ_j (φ')_branch_j^ℓ(ε)

But branch j only carries modes with m ≡ j mod 8.
So the ℓ=0 flux from the trunk must split into only 2 branches (not 3),
because branch 2 doesn't support ℓ=0.

Similarly, the ℓ=1 flux splits into only 2 branches (not 3),
because branch 2 doesn't support ℓ=1 either.

The flux division changes the effective boundary condition for each branch!
""")

def junction_potential_ell(L_vec, ell, eps_j=eps, nu=nu_0, v0=v_UV, v1=v_IR):
    """Compute the on-shell GW action for mode ℓ on the branched geometry.

    The mode ℓ exists on branch j only if j ≤ ℓ.
    (Simplified: branch 0 has all ℓ, branch 1 needs ℓ≥1, branch 2 needs ℓ≥2)

    Matching at junction z = ε:
    - Continuity: φ_trunk(ε) = φ_j(ε) = φ_J for all active branches
    - Flux conservation: φ'_trunk(ε) = Σ_{active j} φ'_j(ε)
    """
    L_tau, L_mu, L_e = L_vec  # L_tau < L_mu < L_e

    # Which branches carry this ℓ mode?
    active_branches = []
    for j in range(3):
        if ell >= j:  # branch j needs ℓ ≥ j
            active_branches.append((j, L_vec[j]))

    if len(active_branches) == 0:
        return 0.0

    n_active = len(active_branches)

    # For given φ_J (junction value), solve each segment
    def flux_residual(phi_J):
        # Trunk [0, eps] with φ(0)=v0, φ(eps)=phi_J
        A_T, B_T = gw_coeffs(0, eps_j, v0, phi_J, nu)
        if A_T is None:
            return 1e10
        flux_trunk = gw_deriv(eps_j, A_T, B_T, nu)

        # Active branches [eps, L_j] with φ(eps)=phi_J, φ(L_j)=v1
        total_branch_flux = 0
        for j, Lj in active_branches:
            if Lj <= eps_j + 0.01:
                return 1e10
            A_j, B_j = gw_coeffs(eps_j, Lj, phi_J, v1, nu)
            if A_j is None:
                return 1e10
            total_branch_flux += gw_deriv(eps_j, A_j, B_j, nu)

        return flux_trunk - total_branch_flux

    # Find junction value φ_J
    try:
        phi_J = brentq(flux_residual, v1 * 0.001, v0 * 100, maxiter=500)
    except:
        # Try broader search
        test_vals = np.linspace(v1 * 0.01, v0 * 50, 1000)
        residuals = [flux_residual(pj) for pj in test_vals]
        # Look for sign change
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i+1] < 0:
                try:
                    phi_J = brentq(flux_residual, test_vals[i], test_vals[i+1])
                    break
                except:
                    continue
        else:
            return 1e10

    # Compute total on-shell action
    V = segment_onshell(0, eps_j, v0, phi_J, nu)
    for j, Lj in active_branches:
        V += segment_onshell(eps_j, Lj, phi_J, v1, nu)

    return V


def total_branched_potential(L_vec, ell_max=3):
    """Total GW potential summing over S² modes.
    Each mode has its own junction equation.
    """
    V_total = 0
    for ell in range(0, ell_max + 1):
        V_ell = junction_potential_ell(L_vec, ell)
        V_total += (2 * ell + 1) * V_ell  # degeneracy factor
    return V_total


# Test: symmetric configuration
L_sym = [L_min, L_min, L_min]
V_sym = total_branched_potential(L_sym, ell_max=3)
print(f"\nSymmetric config (all kL = {L_min:.2f}): V = {V_sym:.6e}")

# Now minimize with asymmetric L values
print("\nMinimizing total branched potential...")

# Parameterize as L_tau = L_base, L_mu = L_base + g1, L_e = L_base + g1 + g2
# Minimize over (L_base, g1, g2) with g1, g2 > 0

def neg_total_potential(params):
    L_base, g1, g2 = params
    if g1 < 0 or g2 < 0 or L_base < 5:
        return 1e10
    L_vec = [L_base, L_base + g1, L_base + g1 + g2]
    return total_branched_potential(L_vec, ell_max=3)

# Try several starting points
best_result = None
best_val = 1e10

for L0_start in [20, 30, 40, 50]:
    for g1_start in [1, 2, 3, 5]:
        for g2_start in [1, 2, 3, 5]:
            x0 = [L0_start, g1_start, g2_start]
            try:
                res = minimize(neg_total_potential, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 0.001, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res
            except:
                pass

if best_result is not None:
    L_base, g1, g2 = best_result.x
    print(f"\nOptimal configuration:")
    print(f"  kL_tau  = {L_base:.4f}")
    print(f"  kL_muon = {L_base + g1:.4f}")
    print(f"  kL_elec = {L_base + g1 + g2:.4f}")
    print(f"  gap(μ→τ) = {g1:.4f}  (target: {gap_mu_tau_obs:.4f})")
    print(f"  gap(e→μ) = {g2:.4f}  (target: {gap_e_mu_obs:.4f})")
    if g1 > 0.001:
        print(f"  gap(μ→τ) / ln(4π) = {g1/ln4pi:.4f}")
    if g2 > 0.001:
        print(f"  gap(e→μ) / ln(4π) = {g2/ln4pi:.4f}")
    if g1 > 0.001 and g2 > 0.001:
        print(f"  ratio (e→μ)/(μ→τ) = {g2/g1:.4f}  (target: {gap_e_mu_obs/gap_mu_tau_obs:.4f})")
else:
    print("Optimization failed.")

# ============================================================
# PART 8: Direct approach — ℓ mode flux splitting
# ============================================================

print("\n" + "=" * 70)
print("PART 7: DIRECT EFFECT — FLUX SPLITTING AT JUNCTION")
print("=" * 70)

print("""
For the ℓ=0 mode:
  Trunk flux splits into 3 branches → each branch gets 1/3 of the flux
  → effective BC: φ'_j = (1/3) × φ'_trunk

For the ℓ=1 mode:
  Only branches 0 and 1 support it → flux splits 2 ways
  Branch 0 and 1 each get 1/2 of the flux
  Branch 2 gets NO ℓ=1 flux

For the ℓ=2 mode:
  All branches support it → flux splits 3 ways

The key difference: branch 2 (electron) receives no ℓ=1 flux but
branches 0 and 1 do. This changes the effective potential for branch 2.

Let's compute the shift from the ℓ=1 flux difference.
""")

# The ℓ=1 mode flux that branch 2 misses:
# On a single funnel, the ℓ=1 contribution to V(L) goes as:
# V_1(L) ∝ (ℓ(ℓ+1)/r₀²) × e^{-4kL} × [correction from angular barrier]

# The simplest estimate: the ℓ=1 mode shifts the GW minimum by:
# δkL ∝ ln(ℓ(ℓ+1)) × (1/something involving the GW parameters)

# More precisely, the one-loop effective potential from the S² KK tower:
# V_1loop = -(1/2) Σ_ℓ (2ℓ+1) × [d/ds det(-∂²_z + ℓ(ℓ+1)/R²)^{-s}]_{s=0}

# The key quantity is the RATIO of determinants for successive ℓ:
# det(-∂² + 2/R²) / det(-∂²) = ?

# On the interval [0, L] with Dirichlet BCs and R(z) = r₀ e^{-kz}:
# The ratio involves the S² eigenvalue 2 (for ℓ=1) vs 0 (for ℓ=0)

# For a CONSTANT potential V on [0, L]:
# det(-d²/dz² + V) = V L / sin(√V × L) ... for Dirichlet BCs
# The ratio det(V=2)/det(V=0) = √2 L / sin(√2 L)

# But our potential is z-dependent: V(z) = ℓ(ℓ+1) e^{2kz}/r₀²
# This grows exponentially, making the ratio much larger near the UV

# At the junction z = ε (UV end), V_ℓ(ε) = ℓ(ℓ+1) e^{2kε}/r₀²
# For ε → 0: V_ℓ(0) = ℓ(ℓ+1)/r₀²

# The S² area at the UV brane: A(S²) = 4π r₀²
# So ℓ(ℓ+1)/r₀² = ℓ(ℓ+1) × 4π / (4π r₀²) = ℓ(ℓ+1) × 4π / A(S²)

# If r₀ = 1 (natural units), the S² area is 4π.
# The contribution of each ℓ mode is proportional to ℓ(ℓ+1)/4π times the S² area.

print("Computing single-funnel minimum with and without ℓ=0, ℓ=1 modes...\n")

# Single funnel potential with specific ℓ modes
def V_single_with_modes(L, ell_list, r0=1.0, weight=0.01):
    """Single funnel GW potential including specified S² modes.

    V = V_tree + weight × Σ_ℓ (2ℓ+1) × V_ℓ(L)

    where V_ℓ is the one-loop correction from the ℓ-th KK mode.
    The weight controls the strength of the one-loop corrections.
    """
    V = V_single(L)

    for ell in ell_list:
        # The ℓ-th mode contributes via the angular barrier
        # Approximate: the shift from the ℓ-th mode is proportional to
        # ℓ(ℓ+1) × e^{2kε}/r₀² × (something involving the GW solution)

        # More careful: integrate the angular barrier term over the GW solution
        A, B = gw_coeffs(0, L, v_UV, v_IR, nu_0)
        if A is None:
            continue

        # ∫₀ᴸ dz e^{-4kz} × ℓ(ℓ+1)e^{2kz}/r₀² × φ²(z)
        # φ(z) = e^{2kz}[A e^{νkz} + B e^{-νkz}]
        # φ²(z) = e^{4kz}[A² e^{2νkz} + 2AB + B² e^{-2νkz}]
        # integral = ℓ(ℓ+1)/r₀² × ∫₀ᴸ dz e^{-4kz+2kz+4kz}[...]
        #          = ℓ(ℓ+1)/r₀² × ∫₀ᴸ dz e^{2kz}[A²e^{2νkz} + 2AB + B²e^{-2νkz}]

        integral = 0
        Nz = 1000
        dz = L / Nz
        for iz in range(Nz):
            z = (iz + 0.5) * dz
            e2kz = exp(2 * z)
            phi_sq_factor = (A**2 * exp(2*nu_0*z) + 2*A*B + B**2 * exp(-2*nu_0*z))
            integral += e2kz * phi_sq_factor * dz

        V_correction = weight * (2 * ell + 1) * ell * (ell + 1) / r0**2 * integral
        V += V_correction

    return V

# Find minima with different mode content
print(f"{'Mode content':<40} {'kL_min':>8} {'Shift':>8}")
print("-" * 60)

ref_L = None
for label, modes, w in [
    ("All ℓ=0,1,2 (tau-like)", [0, 1, 2], 0.01),
    ("ℓ=1,2 only (muon-like, no ℓ=0)", [1, 2], 0.01),
    ("ℓ=2 only (electron-like, no ℓ=0,1)", [2], 0.01),
    ("No S² modes (tree level)", [], 0.01),
]:
    res = minimize_scalar(lambda L: V_single_with_modes(L, modes, weight=w),
                         bounds=(10, 80), method='bounded')
    if ref_L is None:
        ref_L = res.x
    shift = res.x - ref_L
    print(f"  {label:<38} {res.x:>8.4f} {shift:>+8.4f}")

# Try different weights to see if ln(4π) emerges
print(f"\n--- Scanning weight parameter for ln(4π) gap ---")
print(f"Target gap(μ→τ) = {gap_mu_tau_obs:.4f}, gap(e→μ) = {gap_e_mu_obs:.4f}\n")

for w in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    L_tau = minimize_scalar(lambda L: V_single_with_modes(L, [0,1,2], weight=w),
                           bounds=(5, 100), method='bounded').x
    L_mu = minimize_scalar(lambda L: V_single_with_modes(L, [1,2], weight=w),
                          bounds=(5, 100), method='bounded').x
    L_e = minimize_scalar(lambda L: V_single_with_modes(L, [2], weight=w),
                         bounds=(5, 100), method='bounded').x

    g1 = L_mu - L_tau
    g2 = L_e - L_mu

    print(f"  w={w:<6.3f}: kL_τ={L_tau:>7.2f}, kL_μ={L_mu:>7.2f}, kL_e={L_e:>7.2f}  "
          f"gap(μ→τ)={g1:>6.3f}  gap(e→μ)={g2:>6.3f}  ratio={g2/g1 if g1>0.001 else 0:.3f}")

# ============================================================
# PART 9: The S² area mechanism
# ============================================================

print("\n" + "=" * 70)
print("PART 8: THE S² AREA MECHANISM")
print("=" * 70)

print("""
The ln(4π) = ln(Area(S²)) hint.

On a branched manifold, the S² at the junction creates a topological
contribution to the effective potential. When a branch DOESN'T support
a given S² mode, it's as if that branch has a SMALLER effective S².

The effective S² area for each branch:
  Branch 0 (tau):     A_eff = 4π (full S²)
  Branch 1 (muon):    A_eff = 4π - δA₁ (missing ℓ=0 contribution)
  Branch 2 (electron): A_eff = 4π - δA₁ - δA₂ (missing ℓ=0 and ℓ=1)

The GW stabilization length depends on the S² area through:
  kL ∝ (1/δ) × ln(v₀/v₁) + f(A_eff)

If f(A) = (1/2) ln(A), then:
  kL_tau = kL₀ + (1/2) ln(4π)
  kL_muon = kL₀ + (1/2) ln(4π - δA₁)
  kL_electron = kL₀ + (1/2) ln(4π - δA₁ - δA₂)

And the gap:
  gap(μ→τ) = (1/2) [ln(4π) - ln(4π - δA₁)]
  gap(e→μ) = (1/2) [ln(4π - δA₁) - ln(4π - δA₁ - δA₂)]

This gives gap(μ→τ) < gap(e→μ) only if ln is concave... hmm that gives
the WRONG ordering too.

Let me try: each MISSING mode ADDS length to the funnel (lighter particle
= longer funnel = more missing modes pushing the IR brane further out).
""")

# Direct approach: the GW potential depends on the S² determinant
# The determinant contribution from modes 0 to ℓ_max is:
# ln det = Σ_{ℓ=0}^{ℓ_max} (2ℓ+1) × ln[eigenvalue_ℓ(L)]
# where eigenvalue_ℓ(L) is the ℓ-th KK eigenvalue on [0,L]

# The SHIFT in kL from removing mode ℓ:
# δkL_ℓ = (2ℓ+1) × ln[eigenvalue_ℓ(L)] / V''(L)
# where V''(L) is the curvature of the tree-level GW potential

# For the Laplacian on S² of radius r₀:
# eigenvalue_ℓ = ℓ(ℓ+1)/r₀²
# Contribution to S² heat kernel at t: (2ℓ+1) e^{-ℓ(ℓ+1)t}

# The MODE DENSITY on S² up to ℓ=L is:
# N(L) = Σ_{ℓ=0}^{L} (2ℓ+1) = (L+1)²
# For large L: N ~ L² ~ eigenvalue × Area(S²)/(4π)

# The WEYL LAW for S²:
# N(λ) = Area(S²)/(4π) × λ = r₀² λ
# So the density of eigenvalues encodes the S² area.

# Now for the branched geometry:
# Branch 0 sees all modes → effective area = 4π r₀²
# Branch 1 misses ℓ=0 → effective area = 4π r₀² - 4π r₀²/(some factor)
# Branch 2 misses ℓ=0,1 → even smaller effective area

# The NUMBER of missing modes:
# Branch 1 misses: (ℓ,m) = (0,0) → 1 mode
# Branch 2 misses: (0,0), (1,1) → 2 modes [actually (1,2) needs ℓ≥2, so at ℓ=1: m=2>1, not allowed]

# Wait let me recount. For Z₈ with j=2 (electron):
# Need m ≡ 2 mod 8
# ℓ=0: m=0 only, 0≢2 → NO mode
# ℓ=1: m=-1,0,1, none ≡ 2 → NO mode
# ℓ=2: m=-2,-1,0,1,2, m=2 ≡ 2 → YES: (2,2)
# So branch 2 is missing modes at ℓ=0 AND ℓ=1

# For j=1 (muon):
# Need m ≡ 1 mod 8
# ℓ=0: m=0 only, 0≢1 → NO mode
# ℓ=1: m=-1,0,1, m=1 ≡ 1 → YES: (1,1)
# So branch 1 is missing only the ℓ=0 mode

# The S² Casimir energy from the zero mode (ℓ=0):
# E_0 = (1/2) × eigenvalue_0(L) = 0  (ℓ=0 has zero angular eigenvalue)
# This is the CONSTANT mode on S², which has zero angular energy.
# It contributes to the z-equation as the standard GW scalar (no barrier).

# The shift from HAVING vs NOT HAVING the ℓ=0 mode:
# With ℓ=0: the GW potential has an extra attractive term (from the bulk scalar)
# Without ℓ=0: this term is absent → potential is less attractive → IR brane at larger L

# The contribution of the ℓ=0 mode to the potential at the junction:
# V_junction ∝ e^{-4kε} × |φ_J|² × (flux from ℓ=0)

# At the junction, the ℓ=0 mode contributes:
# V₀ = (1/n_branches) × V_single
# where n_branches is the number of branches supporting ℓ=0

# Branch 0 (tau): gets ℓ=0 flux shared with 2 other branches → V₀/3
# Branch 1 (muon): gets ℓ=0 flux shared with 2 other branches → V₀/3
# Branch 2 (electron): gets NO ℓ=0 flux → 0

# Wait, both branches 0 and 1 support ℓ=0. Branch 2 doesn't.
# So ℓ=0 flux splits between branches 0 and 1 only → each gets V₀/2

# Similarly, ℓ=1 flux: branches 0 and 1 support it, branch 2 doesn't
# → splits between branches 0 and 1 → each gets V₁/2

# ℓ=2 flux: all branches support it → splits 3 ways → each gets V₂/3

# Net potential for each branch:
# V_tau   = V₀/2 + V₁/2 + V₂/3 + V₃/3 + ...
# V_muon  = V₀/2 + V₁/2 + V₂/3 + V₃/3 + ...  (SAME as tau for ℓ=0 splitting!)
# V_elec  = 0    + 0    + V₂/3 + V₃/3 + ...

# Hmm, with this counting, tau and muon get the SAME potential.
# That's because both support ℓ=0 and ℓ=1.

# Wait, I need to recheck. For j=1 (muon), allowed m ≡ 1 mod 8:
# ℓ=0: no mode (m=0 only, 0≢1 mod 8)

# So branch 1 (muon) does NOT have ℓ=0 mode!
# Let me redo:
# Branch 0 (tau, j=0): m≡0 mod 8 → ℓ=0: YES (m=0)
# Branch 1 (muon, j=1): m≡1 mod 8 → ℓ=0: NO; ℓ=1: YES (m=1)
# Branch 2 (electron, j=2): m≡2 mod 8 → ℓ=0: NO; ℓ=1: NO; ℓ=2: YES (m=2)

# So:
# ℓ=0: only branch 0 supports it → all ℓ=0 flux goes to branch 0
# ℓ=1: branches 0 and 1 support it → flux splits 2 ways
# ℓ=2: all branches support it → flux splits 3 ways
# ℓ=3: branches 0, 1, 2 all have m≡j with |m|≤3 → splits 3 ways
# etc.

print("\nMode support at junction (corrected):")
print(f"  ℓ=0: branch 0 ONLY → flux ratio 1:0:0")
print(f"  ℓ=1: branches 0,1 → flux ratio 1/2:1/2:0")
print(f"  ℓ=2: branches 0,1,2 → flux ratio 1/3:1/3:1/3")
print(f"  ℓ≥3: all branches → equal split")

print(f"\nNet mode-weighted potential (schematic):")
print(f"  V_tau   = V₀ + V₁/2 + V₂/3 + ...")
print(f"  V_muon  = 0  + V₁/2 + V₂/3 + ...")
print(f"  V_elec  = 0  + 0    + V₂/3 + ...")
print(f"")
print(f"  gap(tau→muon) ~ V₀ (from the ℓ=0 mode that muon doesn't have)")
print(f"  gap(muon→elec) ~ V₁ (from the ℓ=1 mode that electron doesn't have)")
print(f"")
print(f"  V₁ has degeneracy 3 (2ℓ+1=3) vs V₀ with degeneracy 1")
print(f"  So gap(e→μ)/gap(μ→τ) ~ 3/1 × (eigenvalue ratio)")

# The eigenvalue of mode ℓ on S² is ℓ(ℓ+1).
# ℓ=0: eigenvalue 0, degeneracy 1
# ℓ=1: eigenvalue 2, degeneracy 3

# The contribution to the functional determinant:
# δ(ln det) from mode ℓ = (2ℓ+1) × ln(eigenvalue_ℓ/μ²)
# where μ is a regularization scale

# For ℓ=0: eigenvalue = 0 → needs special treatment (zero mode)
# The zero mode on S² contributes ln(Area(S²)) = ln(4π r₀²)
# (From the normalization of the constant mode: 1/√(4πr₀²))

# For ℓ=1: eigenvalue = 2/r₀², degeneracy 3
# Contribution: 3 × ln(2/r₀²)

# The shift in kL from the ℓ=0 zero mode:
# δ(kL) = (1/(2δ)) × ln(4π r₀²) / kL
# This gives ln(4π) if r₀ = 1!

print(f"\n--- Zero mode contribution ---")
r0 = 1.0
print(f"ℓ=0 zero mode normalization: 1/√(4πr₀²) = 1/√(4π) for r₀=1")
print(f"ln(4π r₀²) = ln(4π) = {log(4*pi*r0**2):.6f}")
print(f"")
print(f"If the shift from the ℓ=0 mode is proportional to ln(4π):")
print(f"  gap(μ→τ) = α × ln(4π) where α encodes how the zero mode")
print(f"  enters the GW stabilization equation")

# The ℓ=1 contribution:
# 3 modes with eigenvalue 2/r₀²
# Collective contribution: 3 × ln(2/r₀²) = 3 × ln(2) for r₀=1
print(f"\nℓ=1 contribution: 3 × ln(2) = {3*log(2):.6f}")
print(f"But observed gap(e→μ) = {gap_e_mu_obs:.4f}")
print(f"Ratio gap(e→μ)/gap(μ→τ) observed = {gap_e_mu_obs/gap_mu_tau_obs:.4f}")

# KEY: the ratio should be:
# gap(e→μ)/gap(μ→τ) = (ℓ=1 contribution) / (ℓ=0 contribution)
# = 3 × ln(2) / ln(4π) ???
# = 2.079 / 2.531 = 0.822  → WRONG (should be ~1.89)

# OR: the zero mode contributes ln(4π) to each gap, and the integer
# coefficient comes from the Z₈ pair half-distance (d/2).
# d/2 = 1 for (3,5) → μ→τ → 1×ln(4π)
# d/2 = 2 for (2,6) → e→μ → 2×ln(4π)
# d/2 = 3 for (1,7) → electron anchor → 3×ln(4π) from UV

print(f"\n--- Z₈ pair half-distances as mode multiplicities ---")
print(f"(3,5) pair: d/2 = 1 → 1 zero mode contribution to μ→τ gap")
print(f"(2,6) pair: d/2 = 2 → 2 zero mode contributions to e→μ gap")
print(f"(1,7) pair: d/2 = 3 → electron sits at UV anchor + 3×ln(4π)")
print(f"")
print(f"If each 'zero mode contribution' = ln(4π):")
print(f"  gap(μ→τ) = 1×ln(4π) = {ln4pi:.4f}  (observed: {gap_mu_tau_obs:.4f}, +m_φ/k: {ln4pi+m_phi_over_k:.4f})")
print(f"  gap(e→μ) = 2×ln(4π) = {2*ln4pi:.4f}  (observed: {gap_e_mu_obs:.4f}, +m_φ/k: {2*ln4pi+m_phi_over_k:.4f})")

# ============================================================
# PART 10: Numerical verification — junction GW with mode-dependent flux
# ============================================================

print("\n" + "=" * 70)
print("PART 9: NUMERICAL — MODE-DEPENDENT FLUX SPLITTING")
print("=" * 70)

def V_branch_flux_split(L, j, ell_max=5, r0=1.0, coupling=1.0):
    """GW potential for branch j including mode-dependent flux splitting.

    For mode ℓ:
    - Only branches with Z₈ charge ≤ ℓ support this mode
    - The flux from the trunk splits among supporting branches
    - Branch j gets flux_fraction = 1/(number of supporting branches) if j≤ℓ, else 0

    The effective GW potential for branch j is:
    V_j(L) = Σ_ℓ f_j(ℓ) × (2ℓ+1) × V_ℓ(L)

    where f_j(ℓ) is the fraction of ℓ-flux going to branch j.
    """
    V = V_single(L)  # tree-level (ℓ=0 mode with full flux)

    # Corrections from mode-dependent splitting
    for ell in range(0, ell_max + 1):
        # How many branches support this ℓ?
        n_support = sum(1 for jj in range(3) if jj <= ell)
        n_support = min(n_support, 3)

        # Fraction going to branch j
        if j <= ell:
            f_j = 1.0 / n_support
        else:
            f_j = 0.0

        # The mode contribution (perturbative in the angular barrier)
        if ell > 0:  # ℓ=0 is already in V_single
            # Compute the angular barrier contribution
            A, B = gw_coeffs(0, L, v_UV, v_IR, nu_0)
            if A is None:
                continue

            # ∫₀ᴸ dz e^{-2kz} × φ²(z) with angular weight
            integral = 0
            Nz = 500
            dz = L / Nz
            for iz in range(Nz):
                z = (iz + 0.5) * dz
                phi_val = exp(2*z) * (A * exp(nu_0*z) + B * exp(-nu_0*z))
                integral += exp(-2*z) * phi_val**2 * dz

            V_ell_correction = coupling * f_j * (2*ell+1) * ell*(ell+1) / r0**2 * integral
            V += V_ell_correction
        else:
            # ℓ=0: modify the tree-level by the flux fraction
            # V_single is for 1 branch getting ALL the flux
            # With flux splitting, this branch gets f_j of the flux
            # The potential scales as V ∝ (flux)² approximately
            V = f_j * V_single(L) + (1 - f_j) * V_single(L) * 0  # crude: just reduce V₀
            # Better: the tree-level V depends on the BC, which changes with flux

    return V


# Actually, let me take a completely different approach.
# The flux splitting changes the BOUNDARY CONDITION at the junction.
# This changes the effective v₀ for each branch.

print("\nApproach: Flux splitting changes effective UV VEV for each branch.")
print("")

# In the standard GW, the minimum of V(L) depends on v₀/v₁:
# kL_min = (1/δ) × ln(v₀/v₁)

# If the ℓ=0 mode flux goes entirely to branch 0 (tau),
# then branch 0 has an ENHANCED effective v₀, while branches 1,2 have REDUCED v₀.

# The S² zero mode normalization: φ₀ = 1/√(4π r₀²)
# The ℓ=1 mode normalization: φ₁ = Y₁ₘ with eigenvalue 2/r₀²

# The total scalar VEV at the UV brane decomposes as:
# v₀² = |φ₀|² × 4π r₀² + Σ_{ℓ>0} |φ_ℓ|² × (2ℓ+1)
# The first term gives the zero mode contribution: v₀² × (fraction from ℓ=0)

# For the branched geometry:
# v₀,eff(tau) = v₀ × √(1 + ℓ=0 fraction + ℓ=1 fraction + ...)
# v₀,eff(muon) = v₀ × √(0 + ℓ=1 fraction + ℓ=2 fraction + ...)
# v₀,eff(electron) = v₀ × √(0 + 0 + ℓ=2 fraction + ...)

# The S² mode decomposition of v₀:
# v₀ = Σ_ℓ v₀,ℓ where v₀,ℓ is the component in the ℓ-th S² harmonic
# Total: v₀² = Σ_ℓ (2ℓ+1) v₀,ℓ²

# If v₀ is a constant on S² (s-wave), then only v₀,₀ is nonzero:
# v₀,₀ = v₀ / √(4π r₀²)

# In this case, the ℓ=0 mode carries ALL the VEV.
# Branch 0 gets all of it; branches 1,2 get ZERO VEV from the trunk!

# But that's too extreme. The VEV propagates down each branch independently.
# The junction just determines how the SCALAR FLUX splits.

# Let me go back to the junction equation.
# The junction value φ_J is determined by flux conservation.
# With n branches supporting the mode: φ'_trunk = n × φ'_branch
# (by symmetry, all branches get the same flux if they support the mode)

# The junction equation for ℓ=0 on the branched geometry:
# Trunk: [0, ε] with BCs φ(0) = v₀, φ(ε) = φ_J
# Branch (only branch 0): [ε, L₀] with BCs φ(ε) = φ_J, φ(L₀) = v_IR
# Flux: φ'_trunk(ε) = 1 × φ'_branch(ε)  [only 1 branch]

# vs. for ℓ=2:
# Flux: φ'_trunk(ε) = 3 × φ'_branch(ε)  [3 branches, by symmetry]

# The effective potential for a branch depends on n_branches:
# More branches sharing the flux → each gets less → higher minimum L

# For n branches:
# φ'_trunk(ε) = n × φ'_branch(ε)
# → φ'_branch(ε) = φ'_trunk(ε) / n

# This changes the effective BC at the junction.
# In the single-funnel limit (n=1): φ'_branch = φ'_trunk → no change
# For n=3: φ'_branch = φ'_trunk/3 → reduced gradient →

# The kL shift from flux dilution:
# Recall kL = (1/δ) ln(v₀/v_IR)
# If the effective v₀ is reduced by the flux splitting:
# v₀_eff = v₀ / n^α  (for some power α)
# Then kL_eff = (1/δ) ln(v₀/(n^α × v_IR))
# = kL₀ - (α/δ) ln(n)

# Wait, this shifts kL DOWN (shorter funnel), not up.
# But we need branch 2 (electron) to have the LONGEST funnel.
# The electron has fewer modes → less flux → LESS attractive potential → longer funnel!

# For branch 0 (tau): supports ℓ=0 alone → gets ALL the ℓ=0 flux → deepest potential
# For branch 2 (electron): only supports ℓ≥2 → misses ℓ=0,1 → shallowest potential → longest

# Yes! Fewer modes → less attractive → longer funnel → lighter particle!
# This gives the RIGHT ordering: tau (heaviest, shortest), electron (lightest, longest).

# The gap from the ℓ=0 mode:
# Branch 0 (tau) has ℓ=0 with n=1 (no sharing)
# The ℓ=0 mode contribution to V: V₀
# Branch 1 (muon) doesn't have ℓ=0 → missing V₀
# The shift: ΔkL = V₀ / V''(L*)

# For the GW potential: V(L) ∝ e^{-(4+2δ)kL} (dominant term)
# V'' ∝ (4+2δ)² e^{-(4+2δ)kL}
# V₀ ∝ e^{-(4+2δ)kL} × [S² zero mode factor]

# The S² zero mode factor is related to the S² area: 1/(4π r₀²)
# So the shift ΔkL ∝ 1/(4+2δ)² × 1/(4πr₀²)

# Hmm, this doesn't obviously give ln(4π).

# Let me try yet another angle. The functional determinant approach.

print("--- Functional determinant approach ---")
print()

# On each branch, the one-loop GW potential is:
# V_1loop(L) = -(1/2) ln det[-∂²_z + V_eff(z; L)]
# where V_eff includes the S² mode structure

# The DIFFERENCE in V_1loop between branch 0 (all modes) and branch 1 (no ℓ=0):
# ΔV = -(1/2) ln[det with ℓ=0 / det without ℓ=0]
#     = -(1/2) ln[eigenvalue_0(L)]
# where eigenvalue_0(L) is the ℓ=0 eigenvalue of the z-operator

# For the z-operator -∂² + 4k∂ - m² on [0, L] with Dirichlet BCs:
# The eigenvalues are: λ_n = (nπ/L)² + 4 - m²/k² = (nπ/L)² + 4 - (ν²-4) = (nπ/L)² + 8 - ν²

# Wait, the z-operator for ℓ=0 is the SAME as the standard GW operator.
# Its functional determinant is already included in V_tree.

# The additional modes (ℓ>0) contribute NEW eigenvalue towers:
# For mode ℓ, the z-operator is: -∂² + 4k∂ - m² + ℓ(ℓ+1)e^{2kz}/r₀²
# This has a z-dependent "mass" → no simple eigenvalue formula

# But for the SHIFT in the minimum:
# kL_j = kL_0 + Σ_{ℓ missing} δkL_ℓ
# where δkL_ℓ is the shift from the ℓ-th S² mode

# For the GW potential V(L) = A e^{-(4+2δ)kL} - B e^{-4kL}:
# kL_min = (1/(2δ)) ln[(4+2δ)A/(4B)]

# The one-loop correction shifts this by:
# δkL = [dV_1loop/dL] / [d²V_tree/dL²]|_{L=L_min}

# At the minimum: V' = 0 → -(4+2δ)A e^{-(4+2δ)kL} + 4B e^{-4kL} = 0
# V'' = (4+2δ)² A e^{-(4+2δ)kL} - 16B e^{-4kL}
#      = [(4+2δ)² - 16/(4+2δ) × 4B] × ...
# Let me simplify: at minimum, A e^{-2δkL} = 4B/(4+2δ)
# V'' = A e^{-(4+2δ)kL} × [(4+2δ)² - 16] = A e^{-(4+2δ)kL} × [16+16δ+4δ² - 16]
#      = A e^{-(4+2δ)kL} × (16δ + 4δ²) = A e^{-(4+2δ)kL} × 4δ(4+δ)

# The one-loop potential for the ℓ-th mode (schematic):
# V_1loop,ℓ(L) = -(1/2) × (2ℓ+1) × ln[det_ℓ(L)/det_ℓ(∞)]
# Its L-derivative involves the ℓ-th eigenvalue at the boundary:
# dV_1loop,ℓ/dL = -(1/2) × (2ℓ+1) × d/dL ln det_ℓ(L)

# For large L, det_ℓ(L) ∝ e^{2√V_ℓ × L} where V_ℓ ~ ℓ(ℓ+1)/r₀²
# d/dL ln det_ℓ ∝ 2√(ℓ(ℓ+1))/r₀ → constant for large L

# But at the GW minimum (moderate L), the z-dependent potential matters.

# KEY INSIGHT: For the ℓ=0 mode, the angular barrier is ZERO.
# This is the ONLY mode that propagates all the way to the IR.
# All ℓ>0 modes are exponentially suppressed in the IR by the barrier.

# So the ℓ=0 mode is special: it's the only mode that "sees" the IR brane.
# Removing it from a branch means that branch has NO long-range attraction
# from this mode → the IR brane is pushed further out.

# The shift from removing the ℓ=0 mode:
# This changes the GW potential from N modes to N-1 modes
# The fractional change in V ∝ 1/N_modes at UV
# But the ℓ=0 mode is the ONLY one reaching the IR → it's dominant there

# In the IR (large z), the angular barrier e^{2kz}/r₀² becomes huge.
# Only ℓ=0 survives. So the IR brane is stabilized entirely by ℓ=0.
# Removing ℓ=0 means the IR brane stabilization must come from ℓ=1,2,...
# But these modes are exponentially suppressed in the IR → weaker stabilization → larger L.

# The ℓ=0 stabilization: kL₀ = (1/δ) ln(v₀/v₁)
# The ℓ=1 stabilization: kL₁ = (1/δ₁) ln(v₀/v₁) where δ₁ > δ
# because the angular barrier effectively increases the bulk mass.

# For mode ℓ, the effective ν is:
# ν_eff(ℓ, z) = √(4 + m²/k² + ℓ(ℓ+1)e^{2kz}/r₀²)
# At the UV (z=0): ν_eff = √(ν₀² + ℓ(ℓ+1)/r₀²)

# For r₀ = 1/(2√π) (so that 4πr₀² = 1):
# ν_eff = √(ν₀² + 4π ℓ(ℓ+1))

# For ν₀ = 2.1 (δ=0.1):
# ℓ=1: ν_eff = √(4.41 + 8π) = √(29.54) ≈ 5.44 → δ₁ ≈ 3.44
# kL₁ = (1/3.44) ln(v₀/v₁) ≈ kL₀ × δ/δ₁ = kL₀ × 0.1/3.44 → MUCH shorter!

# This doesn't work for the GW picture either — it makes the wrong prediction.

# Let me try the direct junction computation more carefully.

print("=" * 70)
print("PART 10: CAREFUL JUNCTION COMPUTATION")
print("=" * 70)

# Set up the branched geometry with EXACT matching conditions.
# No perturbative approximation.

# Geometry: trunk [0, ε], three branches [ε, L_j]
# For each ℓ mode:
#   - trunk has the standard GW solution
#   - only branches with j ≤ ℓ support this mode
#   - at junction: continuity + flux conservation (φ'_trunk = Σ_{active} φ'_j)

# For the s-wave (ℓ=0, constant on S²):
# This is the standard GW scalar.
# Active branches: only j=0 (tau)
# Junction: φ'_trunk(ε) = φ'_tau(ε)  [muon and electron don't carry ℓ=0]
# → tau branch behaves like a single funnel [ε, L_tau]
# → muon and electron branches have NO ℓ=0 scalar

# For ℓ=1:
# Active branches: j=0 (tau) and j=1 (muon)
# Junction: φ'_trunk_1(ε) = φ'_tau_1(ε) + φ'_muon_1(ε)

# The total effective potential for each branch:
# V_tau(L_tau)   = V₀(L_tau; n=1) + V₁(L_tau; n=2) + V₂(L_tau; n=3) + ...
# V_muon(L_muon) = V₁(L_muon; n=2) + V₂(L_muon; n=3) + ...
# V_elec(L_elec) = V₂(L_elec; n=3) + ...

# where V_ℓ(L; n) is the ℓ-mode potential with flux split n ways.

# For the ℓ=0 mode with flux split n ways:
# Trunk [0, ε]: φ(0) = v₀, φ(ε) = φ_J
# Each active branch [ε, L]: φ(ε) = φ_J, φ(L) = v₁
# Flux: φ'_trunk(ε) = n × φ'_branch(ε)

# This changes the junction value φ_J.
# For a given L, the junction equation determines φ_J as a function of n.
# Then the on-shell action V(L; n) depends on n through φ_J(n).

def V_mode_n_split(L, n_branches, ell=0, eps_j=eps, nu=nu_0, v0=v_UV, v1=v_IR):
    """GW potential for one branch when flux is split n ways at junction.

    Uses ℓ=0 GW equation (angular barrier handled separately).
    The flux equation: φ'_trunk(ε) = n × φ'_branch(ε)
    """
    if L <= eps_j + 0.01:
        return 1e10

    def flux_eq(phi_J):
        A_T, B_T = gw_coeffs(0, eps_j, v0, phi_J, nu)
        A_B, B_B = gw_coeffs(eps_j, L, phi_J, v1, nu)
        if A_T is None or A_B is None:
            return 1e10

        flux_trunk = gw_deriv(eps_j, A_T, B_T, nu)
        flux_branch = gw_deriv(eps_j, A_B, B_B, nu)

        return flux_trunk - n_branches * flux_branch

    try:
        phi_J = brentq(flux_eq, v1 * 0.001, v0 * 100, maxiter=500)
    except:
        try:
            phi_J = brentq(flux_eq, -v0 * 10, v0 * 100, maxiter=500)
        except:
            return 1e10

    # On-shell action (just the branch part — that's what determines L)
    V = segment_onshell(eps_j, L, phi_J, v1, nu)
    return V


# Find minima for different flux splits
print(f"\n{'Config':<30} {'kL_min':>8} {'V_min':>12}")
print("-" * 55)

kL_mins = {}
for n, label in [(1, 'n=1 (tau ℓ=0)'), (2, 'n=2 (shared ℓ=1)'), (3, 'n=3 (shared ℓ=2)')]:
    res = minimize_scalar(lambda L: V_mode_n_split(L, n), bounds=(5, 100), method='bounded')
    kL_mins[n] = res.x
    print(f"  {label:<28} {res.x:>8.4f} {res.fun:>12.6e}")

if 1 in kL_mins and 2 in kL_mins and 3 in kL_mins:
    shift_12 = kL_mins[2] - kL_mins[1]
    shift_23 = kL_mins[3] - kL_mins[2]
    shift_13 = kL_mins[3] - kL_mins[1]
    print(f"\n  Shift n=1→n=2 (flux dilution): {shift_12:+.4f}")
    print(f"  Shift n=2→n=3 (flux dilution): {shift_23:+.4f}")
    print(f"  Shift n=1→n=3 (flux dilution): {shift_13:+.4f}")
    if abs(shift_12) > 0.001:
        print(f"  Ratio (2→3)/(1→2) = {shift_23/shift_12:.4f}")
    print(f"  Compare: ln(4π) = {ln4pi:.4f}")
    print(f"  Compare: shifts / ln(4π) = {shift_12/ln4pi:.4f}, {shift_23/ln4pi:.4f}")

# ============================================================
# PART 11: Scan over ν (bulk mass parameter)
# ============================================================

print("\n" + "=" * 70)
print("PART 11: PARAMETER SCAN — CAN WE MATCH ln(4π)?")
print("=" * 70)

print(f"\nScanning ν to find where flux-split shifts match observed gaps...")
print(f"Target: gap(μ→τ) = {gap_mu_tau_obs:.4f}, gap(e→μ) = {gap_e_mu_obs:.4f}\n")

# The branched potential model:
# tau   gets ℓ=0 alone (n=1), ℓ=1 shared 2 ways, ℓ=2 shared 3 ways
# muon  gets ℓ=1 shared 2 ways, ℓ=2 shared 3 ways  (no ℓ=0)
# electron gets ℓ=2 shared 3 ways  (no ℓ=0, no ℓ=1)

# Simplified model: effective potential ~ Σ_ℓ (weight_ℓ / n_ℓ) × V_ℓ(L)
# where n_ℓ is the number of branches sharing mode ℓ
# and V_ℓ(L) is the tree-level GW potential (same for all ℓ at tree level)

# The total potential per branch:
# V_tau(L) = V(L; n=1)  [from ℓ=0 being dominant; others are corrections]
# V_muon(L) = V(L; n=2)  [from ℓ=1 being its lowest mode, shared with tau]
# V_elec(L) = V(L; n=3)  [from ℓ=2 being its lowest mode, shared with all]

# (This is a simplification — each branch actually has a sum of modes)

print(f"{'ν':<6} {'kL(n=1)':>9} {'kL(n=2)':>9} {'kL(n=3)':>9} {'gap(1→2)':>9} {'gap(2→3)':>9} {'ratio':>7}")
print("-" * 60)

for nu_test in np.arange(2.01, 2.50, 0.02):
    try:
        kL1 = minimize_scalar(
            lambda L: V_mode_n_split(L, 1, nu=nu_test), bounds=(5, 200), method='bounded'
        ).x
        kL2 = minimize_scalar(
            lambda L: V_mode_n_split(L, 2, nu=nu_test), bounds=(5, 200), method='bounded'
        ).x
        kL3 = minimize_scalar(
            lambda L: V_mode_n_split(L, 3, nu=nu_test), bounds=(5, 200), method='bounded'
        ).x

        g12 = kL2 - kL1
        g23 = kL3 - kL2
        ratio = g23 / g12 if abs(g12) > 0.001 else 0

        print(f"  {nu_test:<5.2f} {kL1:>9.3f} {kL2:>9.3f} {kL3:>9.3f} {g12:>+9.4f} {g23:>+9.4f} {ratio:>7.3f}")
    except:
        pass

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
The branched funnel computation tests whether junction matching conditions
produce mass gaps spaced at ln(4π) ≈ {ln4pi:.4f}.

The Z₈ mode selection creates an asymmetry:
  - tau (j=0): supports all S² modes ℓ ≥ 0
  - muon (j=1): supports ℓ ≥ 1 only
  - electron (j=2): supports ℓ ≥ 2 only

This means the scalar flux at the junction splits differently for each mode:
  - ℓ=0: goes to tau only (n=1)
  - ℓ=1: splits between tau and muon (n=2)
  - ℓ=2: splits among all three (n=3)

Fewer modes → less attractive potential → longer funnel → lighter particle.
The ordering is correct: tau (heaviest) < muon < electron (lightest).

The magnitude of the gaps depends on how much the flux splitting
shifts the GW minimum. The numerical results above show whether
this shift matches ln(4π).

Key targets:
  gap(μ→τ) = {gap_mu_tau_obs:.4f} ≈ ln(4π) + m_φ/k = {ln4pi + m_phi_over_k:.4f}
  gap(e→μ) = {gap_e_mu_obs:.4f} ≈ 2ln(4π) + m_φ/k = {2*ln4pi + m_phi_over_k:.4f}
  ratio    = {gap_e_mu_obs/gap_mu_tau_obs:.4f} ≈ 2.0
""")
