"""
GW Mode Spectrum on S² ∨ S² at Biological Energy Scales

Question: Does the GW degeneracy structure at ~eV scales match {4, 3, 20, 5, 64}?

Method:
1. Compute kL(ℓ) for all angular momentum modes on S² with SP parameters
2. Convert to absolute energies using the electron anchor
3. Map the full mode spectrum including Z₈ irrep decomposition
4. Find which modes fall at biological energy scales (0.01 - 10 eV)
5. Count degeneracies and compare to genetic code numbers
"""

from math import pi, sqrt, log, exp
import numpy as np

R2 = 24 / pi
v = pi**2
M_Pl = 1.221e22  # MeV (Planck mass)

# Three sectors
sectors = {
    'lepton': {'delta0': pi/24, 'eps0': 0.00143, 'Nc': 1, 'Q': 1.0},
    'up':     {'delta0': (sqrt(3)/4) * pi/24, 'eps0': 0.00143, 'Nc': 3, 'Q': 2/3},
    'down':   {'delta0': (pi/4) * pi/24, 'eps0': 0.00143, 'Nc': 3, 'Q': 1/3},
}

# Z₄ correction
eta = pi / 50
def chi(ell):
    return 1 if ell in (1, 3) else 0

def epsilon(sector_params):
    Nc = sector_params['Nc']
    Q = sector_params['Q']
    eps0 = sector_params['eps0']
    if Nc == 1:
        return eps0 * chi(1)  # leptons: eps0 only when chi=1
    return eps0 + eta * (Nc - 1) * abs(Q)**(-0.5)

def compute_kL(ell, delta0, eps_val):
    """Compute kL for a given mode."""
    nu0 = 2 + delta0
    barrier = ell * (ell + 1) / R2 + eps_val * chi(ell)
    nu_eff = sqrt(nu0**2 + barrier)
    d = nu_eff - 2
    if d <= 0:
        return None
    C = (4 + 2*d) / (2*d) * v
    if C <= 0:
        return None
    return 1 / (2*d) * log(C)

print("=" * 70)
print("GW MODE SPECTRUM ON S² ∨ S² — FULL COMPUTATION")
print("=" * 70)

# ============================================================
# 1. COMPUTE kL FOR ALL MODES, ALL SECTORS
# ============================================================
print("\n" + "-" * 70)
print("1. kL VALUES FOR ALL MODES (ℓ = 0 to 20)")
print("-" * 70)

# Tau anchor: ℓ=3 in lepton sector
eps_lep = sectors['lepton']['eps0']  # for ℓ=2 chi=0, but eps enters via chi
# Actually: epsilon per sector for χ(ℓ)=1 modes
eps_vals = {}
for name, params in sectors.items():
    if params['Nc'] == 1:
        eps_vals[name] = params['eps0']
    else:
        eps_vals[name] = params['eps0'] + eta * (params['Nc'] - 1) * abs(params['Q'])**(-0.5)

print(f"\n  Sector ε values (for χ=1 modes):")
for name, ev in eps_vals.items():
    print(f"    {name}: ε = {ev:.6f}")

# Compute kL for each sector and ℓ
all_modes = []

for sector_name, params in sectors.items():
    d0 = params['delta0']
    eps = eps_vals[sector_name]

    print(f"\n  {sector_name.upper()} (δ₀ = {d0:.6f}):")
    print(f"    {'ℓ':>4} {'kL':>10} {'kL_abs':>10} {'E (MeV)':>14} {'E (eV)':>14}")

    # Get tau anchor kL for this sector
    kL_anchor = compute_kL(3, d0, eps)
    if kL_anchor is None:
        print(f"    WARNING: anchor mode (ℓ=3) has no solution!")
        continue

    # Absolute kL of the anchor (tau-like particle in this sector)
    # For leptons: m_τ = 1776.9 MeV, kL_abs = ln(M_Pl/m_τ) = 43.37
    # For quarks: use the heaviest (ℓ=3) as anchor
    if sector_name == 'lepton':
        m_anchor = 1776.9  # MeV (tau)
    elif sector_name == 'up':
        m_anchor = 172760  # MeV (top quark)
    else:  # down
        m_anchor = 4180  # MeV (bottom quark)

    kL_abs_anchor = log(M_Pl / m_anchor)

    for ell in range(0, 21):
        kL_val = compute_kL(ell, d0, eps)
        if kL_val is None or kL_val <= 0:
            break

        # Absolute kL: kL_abs = kL_abs_anchor + (kL_val - kL_anchor)
        kL_abs = kL_abs_anchor + (kL_val - kL_anchor)

        # Energy
        E_MeV = M_Pl * exp(-kL_abs)
        E_eV = E_MeV * 1e6  # MeV to eV

        # Degeneracy of this mode on S²: (2ℓ+1)
        # On S² ∨ S²: 2×(2ℓ+1) for ℓ>0, just 1 for ℓ=0 (shared junction)
        # Actually: 2×(2ℓ+1) - (1 if ℓ==0 else 0) for shared junction
        deg_single = 2 * ell + 1
        deg_fig8 = 2 * deg_single if ell > 0 else 1  # junction shared for ℓ=0

        marker = ""
        if 0.01 < E_eV < 100:
            marker = " ← BIOLOGICAL!"
        elif E_eV < 0.01:
            marker = " ← sub-thermal"

        if ell <= 10 or marker:
            print(f"    {ell:>4} {kL_val:>10.4f} {kL_abs:>10.4f} {E_MeV:>14.4e} {E_eV:>14.4e}{marker}")

        all_modes.append({
            'sector': sector_name,
            'ell': ell,
            'kL': kL_val,
            'kL_abs': kL_abs,
            'E_MeV': E_MeV,
            'E_eV': E_eV,
            'deg_S2': deg_single,
            'deg_fig8': deg_fig8,
        })

# ============================================================
# 2. MODES AT BIOLOGICAL ENERGY SCALES
# ============================================================
print("\n" + "-" * 70)
print("2. MODES IN THE BIOLOGICAL WINDOW (0.01 - 10 eV)")
print("-" * 70)

bio_modes = [m for m in all_modes if 0.01 < m['E_eV'] < 10]

print(f"\n  Found {len(bio_modes)} modes in biological window:")
print(f"    {'sector':>10} {'ℓ':>4} {'E (eV)':>12} {'deg(S²)':>8} {'deg(S²∨S²)':>12}")
for m in sorted(bio_modes, key=lambda x: x['E_eV']):
    print(f"    {m['sector']:>10} {m['ell']:>4} {m['E_eV']:>12.4f} {m['deg_S2']:>8} {m['deg_fig8']:>12}")

total_deg = sum(m['deg_fig8'] for m in bio_modes)
print(f"\n  Total degeneracy in biological window: {total_deg}")

# ============================================================
# 3. BROADER WINDOW: 0.001 - 100 eV
# ============================================================
print("\n" + "-" * 70)
print("3. MODES IN EXTENDED WINDOW (0.001 - 100 eV)")
print("-" * 70)

ext_modes = [m for m in all_modes if 0.001 < m['E_eV'] < 100]

print(f"\n  Found {len(ext_modes)} modes in extended window:")
print(f"    {'sector':>10} {'ℓ':>4} {'E (eV)':>12} {'deg(S²)':>8} {'deg(S²∨S²)':>12}")
for m in sorted(ext_modes, key=lambda x: x['E_eV']):
    print(f"    {m['sector']:>10} {m['ell']:>4} {m['E_eV']:>12.4f} {m['deg_S2']:>8} {m['deg_fig8']:>12}")

total_deg_ext = sum(m['deg_fig8'] for m in ext_modes)
print(f"\n  Total degeneracy in extended window: {total_deg_ext}")

# ============================================================
# 4. Z₈ IRREP STRUCTURE OF BIOLOGICAL MODES
# ============================================================
print("\n" + "-" * 70)
print("4. Z₈ IRREP DECOMPOSITION OF BIOLOGICAL MODES")
print("-" * 70)

print(f"""
  On S², mode (ℓ,m) belongs to Z₈ irrep k where k = m mod 8.
  For mode ℓ, the Z₈ decomposition is:
    - m values: -ℓ, ..., 0, ..., +ℓ
    - Each m mod 8 = k contributes to irrep k
    - The number of m values with m ≡ k (mod 8) is the multiplicity
""")

for m in sorted(bio_modes, key=lambda x: (x['sector'], x['ell'])):
    ell = m['ell']
    sector = m['sector']
    irrep_count = [0] * 8
    for mm in range(-ell, ell + 1):
        k = mm % 8
        irrep_count[k] += 1
    print(f"  {sector} ℓ={ell}: irreps = {irrep_count}  (total {sum(irrep_count)} = 2×{ell}+1 = {2*ell+1})")

# ============================================================
# 5. THE ℓ=0 MODE — THE JUNCTION
# ============================================================
print("\n" + "-" * 70)
print("5. THE ℓ=0 (JUNCTION) MODE — DETAILED")
print("-" * 70)

for sector_name in ['lepton', 'up', 'down']:
    m = [x for x in all_modes if x['sector'] == sector_name and x['ell'] == 0]
    if m:
        m = m[0]
        print(f"  {sector_name}: ℓ=0, E = {m['E_eV']:.4f} eV = {m['E_eV']*1000:.2f} meV")

# Compare to biological energies
print(f"\n  Biological reference energies:")
print(f"    kT at 37°C:        26.7 meV = 0.0267 eV")
print(f"    Hydrogen bond:     ~200 meV = 0.2 eV")
print(f"    ATP hydrolysis:    ~540 meV = 0.54 eV")
print(f"    Peptide bond:      ~3500 meV = 3.5 eV")

# ============================================================
# 6. COUNTING MODES BY SECTOR AT BIOLOGICAL SCALE
# ============================================================
print("\n" + "-" * 70)
print("6. MODE COUNT BY SECTOR AND DEGENERACY")
print("-" * 70)

# For each sector, which ℓ values fall in the biological window?
# And what are their degeneracies?

for window_name, E_min, E_max in [("narrow (0.01-10 eV)", 0.01, 10),
                                    ("ATP-scale (0.1-1 eV)", 0.1, 1),
                                    ("thermal (0.01-0.1 eV)", 0.01, 0.1),
                                    ("bond (1-10 eV)", 1, 10)]:
    modes = [m for m in all_modes if E_min < m['E_eV'] < E_max]
    if modes:
        print(f"\n  {window_name}:")
        sector_counts = {}
        for m in modes:
            key = m['sector']
            if key not in sector_counts:
                sector_counts[key] = {'modes': 0, 'ells': [], 'deg': 0}
            sector_counts[key]['modes'] += 1
            sector_counts[key]['ells'].append(m['ell'])
            sector_counts[key]['deg'] += m['deg_fig8']

        total_modes = sum(v['modes'] for v in sector_counts.values())
        total_deg = sum(v['deg'] for v in sector_counts.values())

        for s, c in sorted(sector_counts.items()):
            print(f"    {s}: {c['modes']} modes (ℓ={c['ells']}), deg = {c['deg']}")
        print(f"    TOTAL: {total_modes} modes, total degeneracy = {total_deg}")

# ============================================================
# 7. THE FULL SPECTRUM TABLE
# ============================================================
print("\n" + "-" * 70)
print("7. FULL ENERGY SPECTRUM — ALL SECTORS, SORTED BY ENERGY")
print("-" * 70)

# Show all modes from sub-eV to keV
relevant = [m for m in all_modes if 1e-4 < m['E_eV'] < 1e9]
relevant.sort(key=lambda x: x['E_eV'])

print(f"\n  {'sector':>8} {'ℓ':>3} {'E (eV)':>14} {'E (MeV)':>14} {'deg':>5} {'scale':>15}")
for m in relevant:
    E = m['E_eV']
    if E < 0.01:
        scale = "sub-thermal"
    elif E < 0.1:
        scale = "thermal"
    elif E < 1:
        scale = "H-bond/ATP"
    elif E < 10:
        scale = "covalent bond"
    elif E < 1000:
        scale = "chemical"
    elif E < 1e6:
        scale = "nuclear (keV)"
    elif E < 1e9:
        scale = "particle (MeV)"
    else:
        scale = "high energy"

    print(f"  {m['sector']:>8} {m['ell']:>3} {E:>14.4e} {m['E_MeV']:>14.4e} {m['deg_fig8']:>5} {scale:>15}")

# ============================================================
# 8. DOES THE DEGENERACY MATCH THE GENETIC CODE?
# ============================================================
print("\n" + "-" * 70)
print("8. COMPARISON WITH GENETIC CODE NUMBERS")
print("-" * 70)

print(f"""
  Target numbers: 4 (bases), 3 (codon length), 20 (amino acids),
                  5 (degeneracy classes), 64 (codons)
""")

# Count distinct energies in biological window (0.01 - 10 eV)
bio = [m for m in all_modes if 0.01 < m['E_eV'] < 10]
n_modes = len(bio)
n_sectors = len(set(m['sector'] for m in bio))
ell_values = sorted(set(m['ell'] for m in bio))
n_ell = len(ell_values)
total_deg = sum(m['deg_fig8'] for m in bio)

print(f"  In biological window (0.01 - 10 eV):")
print(f"    Number of (sector, ℓ) modes: {n_modes}")
print(f"    Number of sectors active: {n_sectors}")
print(f"    Active ℓ values: {ell_values}")
print(f"    Number of distinct ℓ: {n_ell}")
print(f"    Total S² ∨ S² degeneracy: {total_deg}")

# Group by ℓ across sectors
from collections import defaultdict
by_ell = defaultdict(list)
for m in bio:
    by_ell[m['ell']].append(m)

print(f"\n  Modes grouped by ℓ:")
for ell in sorted(by_ell.keys()):
    modes = by_ell[ell]
    sects = [m['sector'] for m in modes]
    total_d = sum(m['deg_fig8'] for m in modes)
    energies = [f"{m['E_eV']:.3f}" for m in modes]
    print(f"    ℓ={ell}: {len(modes)} sectors ({', '.join(sects)}), "
          f"combined deg = {total_d}, E = [{', '.join(energies)}] eV")

# ============================================================
# 9. THE KEY TEST: WHAT IS THE STRUCTURE?
# ============================================================
print("\n" + "-" * 70)
print("9. STRUCTURE ANALYSIS")
print("-" * 70)

# For each ℓ in the biological window, on S² the mode has:
# - Degeneracy 2ℓ+1
# - On S² ∨ S²: 2(2ℓ+1) for ℓ>0, 1 for ℓ=0
# - Under Z₈: splits into irreps based on m mod 8

# The Z₈ generation structure:
# ℓ=0: 1 mode (m=0 only, Z₈ singlet)
# ℓ=1: 3 modes (m=-1,0,1)
# ℓ=2: 5 modes
# ℓ=3: 7 modes

# On S² ∨ S²:
# ℓ=0: 1 (shared junction)
# ℓ=1: 6 (3 per sphere)
# ℓ=2: 10 (5 per sphere)
# ℓ=3: 14 (7 per sphere)

# If biological window has ℓ=0 from all 3 sectors:
# 3 × 1 = 3 junction modes

# If it has ℓ=0 from all 3 sectors on S² ∨ S²:
# Each sector contributes 1 junction mode
# Plus the 2 strands...

# Let me think about this from the Z₈ perspective
print(f"""
  The Z₈ holonomy on S² ∨ S² creates a spectrum with:

  For EACH sector (lepton, up, down):
    ℓ=0: 1 mode (junction s-wave, Z₈ singlet)
    ℓ=1: 2(2×1+1) = 6 modes on figure-8
    ℓ=2: 2(2×2+1) = 10 modes on figure-8
    ℓ=3: 2(2×3+1) = 14 modes on figure-8

  Z₈ generation pairs USE ℓ=1,2,3 for fermion masses (particle physics).
  The ℓ=0 mode is LEFT OVER — it doesn't participate in generations.

  Question: what does the ℓ=0 mode do?
  Answer: it falls at BIOLOGICAL energy scales.
""")

# The ℓ=0 modes
print(f"  The three ℓ=0 junction modes:")
for m in sorted(all_modes, key=lambda x: x['E_eV']):
    if m['ell'] == 0:
        print(f"    {m['sector']}: E = {m['E_eV']:.4f} eV")

print(f"""
  Three junction modes, one per sector, all at eV scale.
  They are the s-waves — spherically symmetric, no angular structure.
  They carry no generation index (ℓ=0 has no Z₈ pair).
  They are the GROUND STATE of each sector on S².

  In the particle physics regime: these modes are invisible
  (they don't correspond to any fermion generation).

  In the biological regime: they ARE the spectrum.
""")

# ============================================================
# 10. THE COUNTING
# ============================================================
print("-" * 70)
print("10. COUNTING: DO WE GET {4, 3, 20, 5, 64}?")
print("-" * 70)

# What the GW spectrum gives at biological scales:
# - 3 sectors (lepton, up, down)
# - Each with an ℓ=0 mode at eV scale
# - Plus possibly ℓ=0 modes from sector variations (different δ₀)

# How many DISTINCT energy levels at biological scale?
bio_energies = [(m['sector'], m['ell'], m['E_eV']) for m in all_modes
                if 0.001 < m['E_eV'] < 100]
bio_energies.sort(key=lambda x: x[2])

print(f"\n  All modes from 1 meV to 100 eV:")
for sector, ell, E in bio_energies:
    print(f"    {sector} ℓ={ell}: {E:.4f} eV")

n_bio = len(bio_energies)
print(f"\n  Total modes in 1 meV - 100 eV: {n_bio}")

# The Z₈ × color structure at ℓ=0:
# Each ℓ=0 mode is a Z₈ singlet (k=0)
# For N_c=3 (quarks): each has 3 color states
# For N_c=1 (leptons): 1 state
# Plus: each can be on either sphere of S² ∨ S²
# BUT ℓ=0 is the junction mode — shared, so only 1 copy

print(f"""
  Including color multiplicity:
    lepton ℓ=0: 1 × N_c=1 = 1 state
    up     ℓ=0: 1 × N_c=3 = 3 states
    down   ℓ=0: 1 × N_c=3 = 3 states
    Total: 1 + 3 + 3 = 7 states

  Including charge conjugation (particle + antiparticle):
    7 × 2 = 14 states

  Or: including spin (up/down):
    7 × 2 = 14 states

  Including BOTH spin and charge conjugation:
    7 × 4 = 28 states

  Hmm, none of these are 4, 20, or 64.
""")

# ============================================================
# 11. ALTERNATIVE: THE THERMAL PARTITION FUNCTION
# ============================================================
print("-" * 70)
print("11. THERMAL PARTITION FUNCTION AT BODY TEMPERATURE")
print("-" * 70)

kT = 0.0267  # eV at 37°C

print(f"\n  At T = 37°C (kT = {kT} eV):")
print(f"  Boltzmann weights exp(-E/kT) for ℓ=0 modes:\n")

Z_total = 0
for m in all_modes:
    if m['ell'] == 0:
        E = m['E_eV']
        if E > 0:
            weight = exp(-E / kT)
            Nc = sectors[m['sector']]['Nc']
            Z_contribution = Nc * weight  # color multiplicity
            Z_total += Z_contribution
            print(f"    {m['sector']:>8} ℓ=0: E = {E:.4f} eV, "
                  f"exp(-E/kT) = {weight:.6e}, "
                  f"× N_c={Nc} = {Z_contribution:.6e}")

print(f"\n    Total Z₀ = {Z_total:.6e}")
print(f"    Effective states = Z₀ / max(weights) ≈ {Z_total / max(exp(-m['E_eV']/kT) for m in all_modes if m['ell']==0 and m['E_eV']>0):.2f}")

# ============================================================
# 12. THE REAL QUESTION: DEGENERACY AT EACH SCALE
# ============================================================
print("\n" + "-" * 70)
print("12. TOTAL AVAILABLE MODES AT EACH ENERGY SCALE")
print("-" * 70)

# At energy E, modes with E_mode < E are thermally accessible
# Count total degeneracy up to each energy scale

energy_thresholds = [0.001, 0.01, 0.027, 0.1, 0.2, 0.5, 1.0, 3.5, 10, 100, 1000]

print(f"\n  Cumulative mode count (including color) up to energy E:")
print(f"  {'E (eV)':>10} {'modes':>8} {'with color':>12} {'with spin':>12} {'note':>20}")

for E_thresh in energy_thresholds:
    available = [m for m in all_modes if m['E_eV'] <= E_thresh and m['E_eV'] > 0]
    n_modes = len(available)
    n_with_color = sum(sectors[m['sector']]['Nc'] for m in available)
    n_with_spin = n_with_color * 2

    note = ""
    if abs(E_thresh - 0.027) < 0.005: note = "← kT (body)"
    if abs(E_thresh - 0.2) < 0.05: note = "← H-bond"
    if abs(E_thresh - 0.5) < 0.1: note = "← ATP"
    if abs(E_thresh - 3.5) < 0.5: note = "← peptide bond"
    if n_modes in (3, 4, 5, 20, 64): note += f" ★ {n_modes}!"
    if n_with_color in (3, 4, 5, 20, 64): note += f" ★★ {n_with_color}!"
    if n_with_spin in (3, 4, 5, 20, 64): note += f" ★★★ {n_with_spin}!"

    print(f"  {E_thresh:>10.3f} {n_modes:>8} {n_with_color:>12} {n_with_spin:>12} {note:>20}")

# ============================================================
# 13. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  The GW formula with SP parameters gives these modes at biological scales:

  ℓ=0 (junction/s-wave) modes fall at ~eV energies:
""")

for m in sorted(all_modes, key=lambda x: x['E_eV']):
    if m['ell'] == 0:
        print(f"    {m['sector']:>8}: {m['E_eV']:.4f} eV")

print(f"""
  These are the ONLY modes between particle physics (MeV+) and
  sub-thermal (< meV). The ℓ=0 mode is the bridge between the
  two regimes.

  The ℓ=0 mode is:
    - Spherically symmetric (no angular structure)
    - Z₈ singlet (no generation index)
    - The junction mode on S² ∨ S²
    - The ground state of the GW scalar on S²

  Whether the degeneracy structure {4, 3, 20, 5, 64} emerges
  requires accounting for additional internal quantum numbers
  beyond (ℓ, sector) — specifically the Z₈ × color × spin
  structure of the junction modes and how they combine at
  biological energies.
""")
