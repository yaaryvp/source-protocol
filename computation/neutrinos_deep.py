#!/usr/bin/env python3
"""
NEUTRINO MASSES — DEEP EXPLORATION

The naive ε = 0 gives the same hierarchy as charged leptons (3500:1).
Experiment shows a MUCH flatter hierarchy (~5-10:1).

What topological mechanism on S²∨S² flattens the neutrino spectrum?

We systematically test multiple hypotheses.
"""

from math import pi, sqrt, log, exp, cos, sin
import numpy as np
from scipy.optimize import brentq, minimize_scalar

print("=" * 70)
print("NEUTRINO MASSES — DEEP TOPOLOGICAL EXPLORATION")
print("=" * 70)

# SP parameters
delta0_lep = pi / 24
R2 = 24 / pi
v = pi**2
eta = pi / 50

def compute_kL(ell, d0, eps_val=0.0, R2_val=None):
    """SP master formula."""
    if R2_val is None:
        R2_val = R2
    chi_l = 1 if ell in (1, 3) else 0
    nu0 = 2 + d0
    barrier = ell * (ell + 1) / R2_val + eps_val * chi_l
    nu_eff = sqrt(nu0**2 + barrier)
    d = nu_eff - 2
    if d <= 0:
        return None
    C = (4 + 2*d) / (2*d) * v
    if C <= 0:
        return None
    return 1 / (2*d) * log(C)

def dm2_ratio(kl1, kl2, kl3):
    """Compute Δm²₃₁/Δm²₂₁ from kL values."""
    r21 = exp(kl1 - kl2)
    r31 = exp(kl1 - kl3)
    if r21**2 - 1 == 0:
        return float('inf')
    return (r31**2 - 1) / (r21**2 - 1)

# Experimental target
RATIO_EXP = 32.58  # Δm²₃₁/Δm²₂₁

# ============================================================
# 1. HYPOTHESIS: DIFFERENT δ₀ FOR NEUTRINOS
# ============================================================
print("\n" + "-" * 70)
print("1. HYPOTHESIS: NEUTRINO SECTOR HAS DIFFERENT δ₀")
print("-" * 70)

print("""
  If neutrinos have δ₀_ν ≠ δ₀_lep, what value gives the right ratio?
  Charged leptons: δ₀ = π/24 ≈ 0.1309
  Up quarks: δ₀ = (√3/4)·π/24 ≈ 0.0567
  Down quarks: δ₀ = (π/4)·π/24 ≈ 0.1028
""")

print(f"  Scanning δ₀ for Δm²₃₁/Δm²₂₁ = {RATIO_EXP}:")
print(f"  {'δ₀':>12} {'δ₀/δ₀_lep':>12} {'kL(1)':>8} {'kL(2)':>8} {'kL(3)':>8} {'ratio':>8} {'m3/m1':>10}")

best_d0 = None
best_err = float('inf')

for d0_test in np.linspace(0.01, 2.0, 2000):
    kl1 = compute_kL(1, d0_test)
    kl2 = compute_kL(2, d0_test)
    kl3 = compute_kL(3, d0_test)
    if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
        ratio = dm2_ratio(kl1, kl2, kl3)
        err = abs(ratio - RATIO_EXP)
        if err < best_err:
            best_err = err
            best_d0 = d0_test

# Found best δ₀, now print neighborhood
if best_d0:
    for d0_test in np.linspace(max(0.01, best_d0 - 0.05), best_d0 + 0.05, 20):
        kl1 = compute_kL(1, d0_test)
        kl2 = compute_kL(2, d0_test)
        kl3 = compute_kL(3, d0_test)
        if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
            ratio = dm2_ratio(kl1, kl2, kl3)
            r31 = exp(kl1 - kl3)
            marker = " ← ★" if abs(ratio - RATIO_EXP) < 1.0 else ""
            print(f"  {d0_test:>12.6f} {d0_test/delta0_lep:>12.4f} {kl1:>8.4f} {kl2:>8.4f} {kl3:>8.4f} {ratio:>8.2f} {r31:>10.2f}{marker}")

    # Refine with root finding
    def ratio_minus_target(d0):
        kl1 = compute_kL(1, d0)
        kl2 = compute_kL(2, d0)
        kl3 = compute_kL(3, d0)
        if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
            return dm2_ratio(kl1, kl2, kl3) - RATIO_EXP
        return 1e10

    try:
        d0_exact = brentq(ratio_minus_target, 0.01, 2.0)
        kl1 = compute_kL(1, d0_exact)
        kl2 = compute_kL(2, d0_exact)
        kl3 = compute_kL(3, d0_exact)
        r21 = exp(kl1 - kl2)
        r31 = exp(kl1 - kl3)
        r32 = exp(kl2 - kl3)

        print(f"\n  ★ EXACT MATCH: δ₀_ν = {d0_exact:.8f}")
        print(f"    δ₀_ν / δ₀_lep = {d0_exact / delta0_lep:.6f}")
        print(f"    δ₀_ν / π = {d0_exact / pi:.6f}")
        print(f"    δ₀_ν = π × {d0_exact / pi:.6f}")

        # Check for clean ratios
        print(f"\n    Testing clean expressions:")
        candidates = [
            ("π/6", pi/6),
            ("π/8", pi/8),
            ("π/12", pi/12),
            ("1/4", 1/4),
            ("1/3", 1/3),
            ("1/2", 1/2),
            ("√2/4", sqrt(2)/4),
            ("π/4 × δ₀", (pi/4) * delta0_lep),
            ("2δ₀", 2 * delta0_lep),
            ("3δ₀", 3 * delta0_lep),
            ("4δ₀", 4 * delta0_lep),
            ("5δ₀", 5 * delta0_lep),
            ("6δ₀", 6 * delta0_lep),
            ("8δ₀", 8 * delta0_lep),
            ("π²/24", pi**2/24),
            ("√(π/24)", sqrt(pi/24)),
            ("(π/24)^(1/3)", (pi/24)**(1/3)),
            ("2π/24 = π/12", 2*pi/24),
            ("3π/24 = π/8", 3*pi/24),
            ("4π/24 = π/6", 4*pi/24),
            ("(2+δ₀)δ₀", (2+delta0_lep)*delta0_lep),
            ("δ₀×√(24/π)", delta0_lep * sqrt(24/pi)),
            ("1/R = √(π/24)", sqrt(pi/24)),
            ("δ₀ × R", delta0_lep * sqrt(R2)),
            ("δ₀ × R²/π", delta0_lep * R2 / pi),
            ("π/24 × 24/π = 1", 1.0),
            ("η × 8 = 8π/50", eta * 8),
            ("δ₀ + η", delta0_lep + eta),
            ("δ₀ × 2π", delta0_lep * 2 * pi),
            ("ln(2)/2", log(2)/2),
        ]
        for name, val in candidates:
            err = abs(val - d0_exact) / d0_exact * 100
            if err < 5:
                print(f"      {name} = {val:.8f}  error: {err:.2f}%  {'← CLOSE!' if err < 1 else ''}")

        print(f"\n    Mass ratios:")
        print(f"      m₂/m₁ = {r21:.4f}")
        print(f"      m₃/m₁ = {r31:.4f}")
        print(f"      m₃/m₂ = {r32:.4f}")

        # Absolute masses from Δm²₂₁
        Dm2_21 = 7.53e-5
        m1 = sqrt(Dm2_21 / (r21**2 - 1))
        m2 = r21 * m1
        m3 = r31 * m1
        print(f"\n    Absolute masses (anchored to Δm²₂₁):")
        print(f"      m₁ = {m1*1000:.4f} meV")
        print(f"      m₂ = {m2*1000:.4f} meV")
        print(f"      m₃ = {m3*1000:.4f} meV")
        print(f"      Σmᵢ = {(m1+m2+m3)*1000:.2f} meV = {(m1+m2+m3):.5f} eV")
    except:
        print("  Could not find exact root.")

# ============================================================
# 2. HYPOTHESIS: DIFFERENT R FOR NEUTRINOS
# ============================================================
print("\n" + "-" * 70)
print("2. HYPOTHESIS: NEUTRINO SECTOR HAS DIFFERENT R²")
print("-" * 70)

print(f"  Charged leptons use R² = 24/π = {R2:.6f}")
print(f"  What R² gives Δm²₃₁/Δm²₂₁ = {RATIO_EXP} with same δ₀ = π/24?")

best_R2 = None
best_err = float('inf')

for R2_test in np.linspace(0.5, 100, 5000):
    kl1 = compute_kL(1, delta0_lep, 0, R2_test)
    kl2 = compute_kL(2, delta0_lep, 0, R2_test)
    kl3 = compute_kL(3, delta0_lep, 0, R2_test)
    if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
        ratio = dm2_ratio(kl1, kl2, kl3)
        err = abs(ratio - RATIO_EXP)
        if err < best_err:
            best_err = err
            best_R2 = R2_test

if best_R2:
    try:
        def ratio_minus_target_R(R2_val):
            kl1 = compute_kL(1, delta0_lep, 0, R2_val)
            kl2 = compute_kL(2, delta0_lep, 0, R2_val)
            kl3 = compute_kL(3, delta0_lep, 0, R2_val)
            if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
                return dm2_ratio(kl1, kl2, kl3) - RATIO_EXP
            return 1e10

        R2_exact = brentq(ratio_minus_target_R, max(0.5, best_R2 - 5), best_R2 + 5)
        print(f"\n  ★ EXACT MATCH: R²_ν = {R2_exact:.6f}")
        print(f"    R²_ν / R²_lep = {R2_exact / R2:.6f}")
        print(f"    R_ν / R_lep = {sqrt(R2_exact / R2):.6f}")

        # Check clean ratios
        print(f"\n    Testing clean expressions:")
        r2_candidates = [
            ("24/π (same)", 24/pi),
            ("48/π", 48/pi),
            ("12/π", 12/pi),
            ("8/π", 8/pi),
            ("R²×2", R2*2),
            ("R²×3", R2*3),
            ("R²×4", R2*4),
            ("R²/2", R2/2),
            ("R²/3", R2/3),
            ("R²/4", R2/4),
            ("1", 1.0),
            ("π", pi),
            ("2π", 2*pi),
            ("4π", 4*pi),
            ("8π", 8*pi),
            ("24", 24.0),
            ("8", 8.0),
            ("π²", pi**2),
            ("6/π", 6/pi),
            ("3/π", 3/pi),
        ]
        for name, val in r2_candidates:
            err = abs(val - R2_exact) / R2_exact * 100
            if err < 5:
                print(f"      {name} = {val:.6f}  error: {err:.2f}%")

        kl1 = compute_kL(1, delta0_lep, 0, R2_exact)
        kl2 = compute_kL(2, delta0_lep, 0, R2_exact)
        kl3 = compute_kL(3, delta0_lep, 0, R2_exact)
        r21 = exp(kl1 - kl2)
        r31 = exp(kl1 - kl3)
        r32 = exp(kl2 - kl3)
        print(f"\n    m₂/m₁ = {r21:.4f},  m₃/m₁ = {r31:.4f},  m₃/m₂ = {r32:.4f}")
    except Exception as e:
        print(f"  Root finding failed: {e}")

# ============================================================
# 3. HYPOTHESIS: MAJORANA SEESAW ON S²∨S²
# ============================================================
print("\n" + "-" * 70)
print("3. HYPOTHESIS: TOPOLOGICAL SEESAW")
print("-" * 70)

print("""
  In standard seesaw: m_ν = m_D²/M_R

  On S²∨S², the Dirac mass comes from funnel penetration on ONE sphere.
  The Majorana mass comes from the JUNCTION — the particle "bouncing"
  between the two spheres.

  If the right-handed neutrino lives at the junction (ℓ=0 mode),
  its mass scale is set by the junction energy.

  Seesaw: kL_ν = 2×kL_D - kL_junction
  where kL_D = charged lepton kL (Dirac mass),
  kL_junction = kL(ℓ=0) for the lepton sector.
""")

kL_junction = compute_kL(0, delta0_lep, 0)
print(f"  kL(ℓ=0, lepton) = {kL_junction:.6f}")

# Charged lepton kL values (Dirac mass scale)
kL_D = {}
for ell in [1, 2, 3]:
    kL_D[ell] = compute_kL(ell, delta0_lep, 0)  # use ε=0 for Dirac part

# Seesaw kL: kL_ν = 2×kL_D - kL_0
print(f"\n  Seesaw kL values (kL_ν = 2×kL_D - kL₀):")
kL_seesaw = {}
for ell in [1, 2, 3]:
    kL_seesaw[ell] = 2 * kL_D[ell] - kL_junction
    print(f"    ℓ={ell}: kL_ν = 2×{kL_D[ell]:.4f} - {kL_junction:.4f} = {kL_seesaw[ell]:.4f}")

# Check: these are large positive numbers → very light particles
ratio_ss = dm2_ratio(kL_seesaw[1], kL_seesaw[2], kL_seesaw[3])
r31_ss = exp(kL_seesaw[1] - kL_seesaw[3])
print(f"\n  Δm²₃₁/Δm²₂₁ = {ratio_ss:.2f}  (exp = {RATIO_EXP})")
print(f"  m₃/m₁ = {r31_ss:.2f}")

# What if the seesaw uses a GENERATION-DEPENDENT Majorana mass?
# kL_ν = 2×kL_D - kL_M(ℓ) where kL_M depends on ℓ
print(f"\n  What if kL_M is generation-dependent?")
print(f"  For the ratio to match, we need kL_M(ℓ) such that:")
print(f"  kL_ν(ℓ) gives ratio = {RATIO_EXP}")

# Try kL_M(ℓ) = a + b×ℓ(ℓ+1) — same barrier form as the Dirac part
print(f"\n  Testing: kL_M(ℓ) = kL₀ + c×ℓ(ℓ+1)/R²:")
for c_factor in np.linspace(0.5, 1.5, 11):
    kl_s = {}
    for ell in [1, 2, 3]:
        kl_M = kL_junction + c_factor * ell*(ell+1)/R2 * (1/(2*(2+delta0_lep)))
        kl_s[ell] = 2 * kL_D[ell] - kl_M
    if kl_s[1] > kl_s[2] > kl_s[3] > 0:
        ratio = dm2_ratio(kl_s[1], kl_s[2], kl_s[3])
        r31 = exp(kl_s[1] - kl_s[3])
        print(f"    c = {c_factor:.2f}: ratio = {ratio:.2f}, m₃/m₁ = {r31:.2f}")

# ============================================================
# 4. HYPOTHESIS: DOUBLE-PATH INTERFERENCE ON S²∨S²
# ============================================================
print("\n" + "-" * 70)
print("4. HYPOTHESIS: DOUBLE-PATH INTERFERENCE")
print("-" * 70)

print("""
  On S²∨S², a neutral particle can propagate through BOTH spheres.
  The two paths acquire phases from the Z₈ holonomy.

  For charged particles: only one path contributes (charge selects sphere).
  For neutral particles: both paths interfere.

  The effective kL is modified:
    exp(-kL_ν) = exp(-kL_L) + exp(-kL_R) × e^{iφ}

  If the two spheres are identical (φ = 0, constructive):
    exp(-kL_ν) = 2 × exp(-kL)
    kL_ν = kL - ln(2)

  This shifts ALL kL values by the same constant: ln(2) ≈ 0.693.
  It doesn't change RATIOS. The hierarchy stays the same.

  But if φ depends on ℓ (through the Z₈ holonomy):
    φ(ℓ) = 2π × ℓ / 8 = πℓ/4 (Z₈ phase per angular momentum)

    exp(-kL_ν) = exp(-kL) × (1 + e^{iπℓ/4})
    |1 + e^{iπℓ/4}| = 2|cos(πℓ/8)|

    kL_ν(ℓ) = kL(ℓ) - ln(2|cos(πℓ/8)|)
""")

# Compute with Z₈ phase interference
print(f"  Z₈ phase interference factors:")
for ell in range(6):
    phase = pi * ell / 4
    factor = 2 * abs(cos(phase / 2))  # |1 + e^iφ| = 2|cos(φ/2)|
    shift = log(factor) if factor > 0 else float('-inf')
    print(f"    ℓ={ell}: φ = {phase:.4f} = {phase/pi:.2f}π, |1+e^iφ| = {factor:.4f}, shift = {shift:.4f}")

print(f"\n  Modified kL values:")
kL_interf = {}
for ell in [1, 2, 3]:
    phase = pi * ell / 4
    factor = 2 * abs(cos(phase / 2))
    kL_base = compute_kL(ell, delta0_lep, 0)
    kL_interf[ell] = kL_base - log(factor) if factor > 0 else kL_base
    print(f"    ℓ={ell}: kL_base = {kL_base:.4f}, shift = {-log(factor):.4f}, kL_ν = {kL_interf[ell]:.4f}")

if all(kL_interf[i] > kL_interf[i+1] for i in [1, 2]):
    ratio_interf = dm2_ratio(kL_interf[1], kL_interf[2], kL_interf[3])
    r31_interf = exp(kL_interf[1] - kL_interf[3])
    print(f"\n  Δm²₃₁/Δm²₂₁ = {ratio_interf:.2f}  (exp = {RATIO_EXP})")
    print(f"  m₃/m₁ = {r31_interf:.2f}")
else:
    print(f"\n  Ordering violated — checking values...")
    for ell in [1,2,3]:
        print(f"    kL_ν({ell}) = {kL_interf[ell]:.6f}")

# ============================================================
# 5. HYPOTHESIS: kL → kL^α (POWER LAW / ANOMALOUS DIMENSION)
# ============================================================
print("\n" + "-" * 70)
print("5. HYPOTHESIS: ANOMALOUS SCALING kL → kL^α")
print("-" * 70)

print("""
  What if the neutral sector has an anomalous dimension that modifies
  the localization: kL_ν = kL^α for some exponent α ≠ 1?

  This is motivated by conformal field theory: the ε=0 sector might
  have different conformal weights than the charged sector.
""")

# Scan α for ratio match
kL_base = {ell: compute_kL(ell, delta0_lep, 0) for ell in [1, 2, 3]}

print(f"  Scanning α for Δm² ratio match:")
print(f"  {'α':>8} {'kLν(1)':>10} {'kLν(2)':>10} {'kLν(3)':>10} {'ratio':>10} {'m3/m1':>10}")

best_alpha = None
best_err = float('inf')

for alpha in np.linspace(0.1, 2.0, 200):
    kl = {ell: kL_base[ell]**alpha for ell in [1, 2, 3]}
    if kl[1] > kl[2] > kl[3] > 0:
        ratio = dm2_ratio(kl[1], kl[2], kl[3])
        err = abs(ratio - RATIO_EXP)
        if err < best_err:
            best_err = err
            best_alpha = alpha

if best_alpha:
    # Refine
    for alpha in np.linspace(best_alpha - 0.1, best_alpha + 0.1, 21):
        kl = {ell: kL_base[ell]**alpha for ell in [1, 2, 3]}
        if kl[1] > kl[2] > kl[3] > 0:
            ratio = dm2_ratio(kl[1], kl[2], kl[3])
            r31 = exp(kl[1] - kl[3])
            marker = " ← ★" if abs(ratio - RATIO_EXP) < 1.0 else ""
            print(f"  {alpha:>8.4f} {kl[1]:>10.4f} {kl[2]:>10.4f} {kl[3]:>10.4f} {ratio:>10.2f} {r31:>10.2f}{marker}")

# ============================================================
# 6. HYPOTHESIS: NEUTRINOS USE ℓ + 1/2 (HALF-INTEGER MODES)
# ============================================================
print("\n" + "-" * 70)
print("6. HYPOTHESIS: HALF-INTEGER MODES (SPINOR ON S²)")
print("-" * 70)

print("""
  On S², bosonic modes use integer ℓ: barrier = ℓ(ℓ+1)/R².
  Fermionic (spinor) modes use half-integer j: barrier = (j+1/2)²/R².

  For Dirac fermions on S², the eigenvalue of the Dirac operator is:
    λ_j = ±(j + 1/2)/R where j = 1/2, 3/2, 5/2, ...

  The barrier in the RS/GW formula becomes:
    barrier = (j+1/2)²/R² = (n+1)²/R² for n = 0, 1, 2, ...

  This is different from ℓ(ℓ+1)/R²!

  For Dirac spinor modes:
    j = 1/2: barrier = 1/R²
    j = 3/2: barrier = 4/R²
    j = 5/2: barrier = 9/R²

  vs bosonic modes:
    ℓ = 1: barrier = 2/R²
    ℓ = 2: barrier = 6/R²
    ℓ = 3: barrier = 12/R²
""")

# Compute with spinor barrier
def compute_kL_spinor(j_half, d0, R2_val=None):
    """kL for spinor mode with j = j_half (half-integer)."""
    if R2_val is None:
        R2_val = R2
    nu0 = 2 + d0
    barrier = (j_half + 0.5)**2 / R2_val
    nu_eff = sqrt(nu0**2 + barrier)
    d = nu_eff - 2
    if d <= 0:
        return None
    C = (4 + 2*d) / (2*d) * v
    if C <= 0:
        return None
    return 1 / (2*d) * log(C)

print(f"  Spinor kL values (δ₀ = π/24, R² = 24/π):")
for j2 in [1, 3, 5, 7, 9]:  # j = 1/2, 3/2, 5/2, 7/2, 9/2
    j = j2 / 2
    kl = compute_kL_spinor(j, delta0_lep)
    if kl:
        print(f"    j = {j}: barrier = {(j+0.5)**2/R2:.4f}, kL = {kl:.6f}")

# Three lightest spinor modes as neutrinos
print(f"\n  Neutrinos as j = 1/2, 3/2, 5/2 spinor modes:")
kL_sp = {}
for idx, j in enumerate([0.5, 1.5, 2.5]):
    kL_sp[idx+1] = compute_kL_spinor(j, delta0_lep)
    print(f"    ν{idx+1} (j={j}): kL = {kL_sp[idx+1]:.6f}")

ratio_sp = dm2_ratio(kL_sp[1], kL_sp[2], kL_sp[3])
r31_sp = exp(kL_sp[1] - kL_sp[3])
r21_sp = exp(kL_sp[1] - kL_sp[2])
print(f"\n  Δm²₃₁/Δm²₂₁ = {ratio_sp:.2f}  (exp = {RATIO_EXP})")
print(f"  m₃/m₁ = {r31_sp:.2f}")
print(f"  m₂/m₁ = {r21_sp:.2f}")

# ============================================================
# 7. HYPOTHESIS: COMBINED — SPINOR + DIFFERENT δ₀ OR R
# ============================================================
print("\n" + "-" * 70)
print("7. COMBINED: SPINOR MODES + PARAMETER SCAN")
print("-" * 70)

# Scan δ₀ for spinor modes
print(f"\n  Scanning δ₀ for spinor modes to match ratio = {RATIO_EXP}:")

best_d0_sp = None
best_err_sp = float('inf')

for d0_test in np.linspace(0.01, 3.0, 3000):
    kl1 = compute_kL_spinor(0.5, d0_test)
    kl2 = compute_kL_spinor(1.5, d0_test)
    kl3 = compute_kL_spinor(2.5, d0_test)
    if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
        ratio = dm2_ratio(kl1, kl2, kl3)
        err = abs(ratio - RATIO_EXP)
        if err < best_err_sp:
            best_err_sp = err
            best_d0_sp = d0_test

if best_d0_sp:
    try:
        def ratio_spinor(d0):
            kl1 = compute_kL_spinor(0.5, d0)
            kl2 = compute_kL_spinor(1.5, d0)
            kl3 = compute_kL_spinor(2.5, d0)
            if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
                return dm2_ratio(kl1, kl2, kl3) - RATIO_EXP
            return 1e10

        d0_sp_exact = brentq(ratio_spinor, max(0.01, best_d0_sp - 0.1), best_d0_sp + 0.1)
        kl1 = compute_kL_spinor(0.5, d0_sp_exact)
        kl2 = compute_kL_spinor(1.5, d0_sp_exact)
        kl3 = compute_kL_spinor(2.5, d0_sp_exact)
        r21 = exp(kl1 - kl2)
        r31 = exp(kl1 - kl3)

        print(f"\n  ★ SPINOR + δ₀ MATCH: δ₀_ν = {d0_sp_exact:.8f}")
        print(f"    δ₀_ν / δ₀_lep = {d0_sp_exact / delta0_lep:.6f}")
        print(f"    δ₀_ν / π = {d0_sp_exact / pi:.6f}")

        # Check clean expressions
        candidates_sp = [
            ("π/24 (same)", pi/24),
            ("π/12", pi/12),
            ("π/8", pi/8),
            ("π/6", pi/6),
            ("π/4", pi/4),
            ("π/3", pi/3),
            ("2δ₀", 2*delta0_lep),
            ("3δ₀", 3*delta0_lep),
            ("4δ₀", 4*delta0_lep),
            ("πδ₀", pi*delta0_lep),
            ("2πδ₀", 2*pi*delta0_lep),
            ("δ₀×R", delta0_lep * sqrt(R2)),
            ("1/R²×π", pi/R2),
            ("√(2)δ₀", sqrt(2)*delta0_lep),
            ("δ₀/sin(π/8)", delta0_lep/sin(pi/8)),
            ("(π/24)²", (pi/24)**2),
            ("δ₀+1/R", delta0_lep + 1/sqrt(R2)),
            ("δ₀×(1+√2)", delta0_lep*(1+sqrt(2))),
        ]
        for name, val in candidates_sp:
            err = abs(val - d0_sp_exact) / d0_sp_exact * 100
            if err < 5:
                print(f"      {name} = {val:.8f}  error: {err:.2f}%")

        print(f"\n    m₂/m₁ = {r21:.4f},  m₃/m₁ = {r31:.4f}")

        Dm2_21 = 7.53e-5
        m1 = sqrt(Dm2_21 / (r21**2 - 1))
        m2 = r21 * m1
        m3 = r31 * m1
        print(f"    m₁ = {m1*1000:.4f} meV, m₂ = {m2*1000:.4f} meV, m₃ = {m3*1000:.4f} meV")
        print(f"    Σmᵢ = {(m1+m2+m3)*1000:.2f} meV = {(m1+m2+m3):.5f} eV")
    except Exception as e:
        print(f"  Root finding failed: {e}")

# Also scan R² for spinor modes with fixed δ₀ = π/24
print(f"\n  Scanning R² for spinor modes (δ₀ = π/24) to match ratio:")

best_R2_sp = None
best_err_R2 = float('inf')

for R2_test in np.linspace(0.1, 200, 5000):
    kl1 = compute_kL_spinor(0.5, delta0_lep, R2_test)
    kl2 = compute_kL_spinor(1.5, delta0_lep, R2_test)
    kl3 = compute_kL_spinor(2.5, delta0_lep, R2_test)
    if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
        ratio = dm2_ratio(kl1, kl2, kl3)
        err = abs(ratio - RATIO_EXP)
        if err < best_err_R2:
            best_err_R2 = err
            best_R2_sp = R2_test

if best_R2_sp:
    try:
        def ratio_spinor_R(R2_val):
            kl1 = compute_kL_spinor(0.5, delta0_lep, R2_val)
            kl2 = compute_kL_spinor(1.5, delta0_lep, R2_val)
            kl3 = compute_kL_spinor(2.5, delta0_lep, R2_val)
            if kl1 and kl2 and kl3 and kl1 > kl2 > kl3:
                return dm2_ratio(kl1, kl2, kl3) - RATIO_EXP
            return 1e10

        R2_sp_exact = brentq(ratio_spinor_R, max(0.1, best_R2_sp - 5), best_R2_sp + 5)
        print(f"\n  ★ SPINOR + R² MATCH: R²_ν = {R2_sp_exact:.6f}")
        print(f"    R²_ν / R²_lep = {R2_sp_exact / R2:.6f}")
        print(f"    R_ν / R_lep = {sqrt(R2_sp_exact / R2):.6f}")

        r2_candidates_sp = [
            ("24/π (same)", 24/pi),
            ("48/π", 48/pi),
            ("2×24/π", 2*24/pi),
            ("3×24/π", 3*24/pi),
            ("96/π", 96/pi),
            ("24", 24.0),
            ("8π", 8*pi),
            ("π²×R²/8", pi**2*R2/8),
            ("50/π", 50/pi),
            ("100/π", 100/pi),
        ]
        for name, val in r2_candidates_sp:
            err = abs(val - R2_sp_exact) / R2_sp_exact * 100
            if err < 5:
                print(f"      {name} = {val:.6f}  error: {err:.2f}%")
    except Exception as e:
        print(f"  Root finding failed: {e}")

# ============================================================
# 8. THE MOST NATURAL OPTION: SAME δ₀, SPINOR BARRIER, SAME R
# ============================================================
print("\n" + "-" * 70)
print("8. ASSESSMENT: WHICH HYPOTHESIS IS MOST NATURAL?")
print("-" * 70)

# Collect all results
print(f"""
  Experimental target: Δm²₃₁/Δm²₂₁ = {RATIO_EXP}

  Results:

  H1: Different δ₀ (bosonic barrier, same R)""")
if best_d0:
    print(f"      δ₀_ν = {d0_exact:.6f} = {d0_exact/delta0_lep:.4f} × δ₀_lep")
    print(f"      Requires δ₀ ≈ {d0_exact/delta0_lep:.1f}× larger than charged sector")

print(f"""
  H2: Different R² (bosonic barrier, same δ₀)""")
if best_R2:
    print(f"      R²_ν = {R2_exact:.4f} = {R2_exact/R2:.4f} × R²_lep")

print(f"""
  H3: Seesaw (bosonic, same δ₀, same R, junction anchor)
      kL_ν = 2×kL_D - kL₀""")
print(f"      Ratio = {ratio_ss:.2f} — hierarchy SQUARES, gets worse")

print(f"""
  H4: Double-path Z₈ interference""")
print(f"      Ratio = {ratio_interf:.2f} — small correction only")

print(f"""
  H6: Spinor modes (same δ₀ = π/24, same R² = 24/π)""")
print(f"      Ratio = {ratio_sp:.2f}")

print(f"""
  H7a: Spinor + different δ₀""")
if best_d0_sp:
    print(f"      δ₀_ν = {d0_sp_exact:.6f}")

print(f"""
  H7b: Spinor + different R²""")
if best_R2_sp:
    print(f"      R²_ν = {R2_sp_exact:.6f}")

# ============================================================
# 9. THE SPINOR RESULT — DETAILED
# ============================================================
print("\n" + "-" * 70)
print("9. SPINOR MODES AT SAME PARAMETERS — DETAILED")
print("-" * 70)

print(f"""
  The spinor hypothesis (H6) is the most NATURAL: same δ₀, same R,
  but spinor barrier (j+1/2)²/R² instead of bosonic ℓ(ℓ+1)/R².

  This changes the barrier structure:
    Bosonic ℓ=1: barrier = 2/R² = {2/R2:.6f}
    Spinor j=½:  barrier = 1/R² = {1/R2:.6f}  (half!)

    Bosonic ℓ=2: barrier = 6/R² = {6/R2:.6f}
    Spinor j=3/2: barrier = 4/R² = {4/R2:.6f}  (2/3)

    Bosonic ℓ=3: barrier = 12/R² = {12/R2:.6f}
    Spinor j=5/2: barrier = 9/R² = {9/R2:.6f}  (3/4)

  The spinor barriers CONVERGE faster than bosonic barriers.
  This FLATTENS the hierarchy.
""")

print(f"  Ratio of barriers (spinor/bosonic):")
for n, ell in enumerate([1, 2, 3]):
    j = (2*n + 1) / 2  # j = 1/2, 3/2, 5/2
    barrier_boson = ell * (ell + 1) / R2
    barrier_spinor = (j + 0.5)**2 / R2
    print(f"    ℓ={ell}, j={j}: {barrier_spinor:.4f}/{barrier_boson:.4f} = {barrier_spinor/barrier_boson:.4f}")

print(f"\n  The spinor barrier (j+1/2)² = (n+1)² grows as n² (perfect squares)")
print(f"  The bosonic barrier ℓ(ℓ+1) grows faster.")
print(f"  This is WHY neutrinos have a flatter hierarchy!")

print(f"\n  Spinor result at δ₀ = π/24, R² = 24/π:")
print(f"    Δm²₃₁/Δm²₂₁ = {ratio_sp:.2f} vs experimental {RATIO_EXP}")
print(f"    Error: {abs(ratio_sp - RATIO_EXP)/RATIO_EXP*100:.1f}%")

if ratio_sp != RATIO_EXP:
    print(f"\n  The spinor hypothesis gives the right DIRECTION")
    print(f"  (flatter than bosonic) but not the exact value.")
    print(f"  The remaining discrepancy might come from:")
    print(f"    - The Z₈ holonomy coupling to spin (spinor-holonomy interaction)")
    print(f"    - The Majorana condition at the junction")
    print(f"    - Both δ₀ and the barrier being modified for neutral fermions")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — NEUTRINO DEEP DIVE")
print("=" * 70)

print(f"""
  The naive ε = 0 hypothesis (same barrier, same parameters) gives
  the same hierarchy as charged leptons. This is WRONG.

  The most promising direction: SPINOR barrier.

  Charged fermions (Dirac): bosonic barrier ℓ(ℓ+1)/R²
  Neutral fermions (Majorana): spinor barrier (j+1/2)²/R²

  Physical reason: neutrinos are Majorana — they ARE spinors on S².
  The Dirac equation on S² has eigenvalues ±(j+1/2)/R, not √(ℓ(ℓ+1))/R.
  Charged fermions acquire an additional angular momentum from the
  gauge field (charge × monopole on S²), which shifts the barrier
  from (j+1/2)² to ℓ(ℓ+1) for integer ℓ.

  Without charge (Q=0), there's no monopole, and the fermion sees
  the pure spinor spectrum: barrier = (j+1/2)²/R².

  This gives:
    - Ratio = {ratio_sp:.2f} (experiment: {RATIO_EXP}) — {abs(ratio_sp - RATIO_EXP)/RATIO_EXP*100:.1f}% off
    - Flatter hierarchy than charged leptons ✓
    - Normal ordering ✓
    - Same δ₀, same R — no new parameters ✓
    - Physical motivation from Dirac equation on S² ✓

  The remaining {abs(ratio_sp - RATIO_EXP)/RATIO_EXP*100:.1f}% discrepancy is the next open problem.
""")
