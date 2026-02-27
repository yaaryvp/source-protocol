"""
Quick check: does SP structure appear in biology?

The genetic code: 64 codons → 20 amino acids + stops
DNA: two strands (S² ∨ S²), base pairing (A↔T, C↔G)
Codons: triplets (3 = generations?)

Key SP numbers to look for:
  3 (generations), 8 (Z₈), 24 (Z₈×3), 50 (2×5²)
  Ratios: 1.889 (gap ratio), √3/4, π/4, π/50
"""

from math import pi, sqrt, log, exp
import numpy as np

print("=" * 70)
print("SP STRUCTURE IN THE GENETIC CODE")
print("=" * 70)

# ============================================================
# 1. THE BASIC NUMBERS
# ============================================================
print("\n" + "-" * 70)
print("1. COUNTING")
print("-" * 70)

print(f"""
  DNA bases: 4 (A, T, C, G)
  Base pairs: 2 (A↔T, C↔G) — Watson-Crick complementarity
  Codon length: 3 (triplets)
  Total codons: 4³ = 64
  Amino acids: 20
  Stop codons: 3
  Coding codons: 61

  SP numbers:
    Z₈ order: 8 = 2 × 4 bases
    Z₄ = Z₈/Z₂: 4 = number of bases!
    3 generations = codon length = stop codons
    64 = 8² = Z₈ × Z₈ (two strands?)
    64 = 4³ = Z₄³ (three generations of Z₄)
""")

# ============================================================
# 2. WHY 20 AMINO ACIDS?
# ============================================================
print("-" * 70)
print("2. WHY 20 AMINO ACIDS?")
print("-" * 70)

# SP mode counts:
# Z₈: 5 independent sectors (j=0..4), 3 pairs + 2 singlets
# If genetic code has Z₄ symmetry (4 bases):
# Z₄: elements 0,1,2,3. Pairing j ↔ 4-j:
#   (0): singlet, (1,3): pair, (2): singlet
# Independent sectors: 3 (j=0,1,2)

print(f"  Z₄ independent sectors: 3 (j=0,1,2)")
print(f"    j=0: singlet")
print(f"    j=1, j=3: pair (1 generation)")
print(f"    j=2: singlet")

# For codons (length 3), the symmetry group acts on each position
# Three positions × Z₄ symmetry = Z₄ × Z₄ × Z₄ / codon equivalences

# The number of orbits of Z₄ acting on {0,1,2,3}³:
# By Burnside's lemma...
# But actually, the question is simpler.

# Watson-Crick: complementary bases pair. This is a Z₂ symmetry.
# A↔T (purine ↔ pyrimidine), C↔G (pyrimidine ↔ purine)
# The complement reversal: read the other strand 3'→5'
# A codon and its reverse complement code for... different amino acids generally

# Key observation: the degeneracy pattern
codon_table = {
    'Phe': 2, 'Leu': 6, 'Ile': 3, 'Met': 1, 'Val': 4,
    'Ser': 6, 'Pro': 4, 'Thr': 4, 'Ala': 4, 'Tyr': 2,
    'His': 2, 'Gln': 2, 'Asn': 2, 'Lys': 2, 'Asp': 2,
    'Glu': 2, 'Cys': 2, 'Trp': 1, 'Arg': 6, 'Gly': 4,
    'Stop': 3
}

# Count by degeneracy
from collections import Counter
deg_counts = Counter(codon_table.values())
print(f"\n  Codon degeneracy distribution:")
print(f"    {'codons':>8} {'amino acids':>12}")
for d in sorted(deg_counts.keys()):
    items = [k for k,v in codon_table.items() if v == d]
    print(f"    {d:>8} {deg_counts[d]:>12}  ({', '.join(items)})")

total_aa = sum(1 for k in codon_table if k != 'Stop')
print(f"\n  Total amino acids: {total_aa}")
print(f"  Total with stop: {total_aa + 1} (if 3 stops count as 1 signal)")

# SP predictions for 20:
print(f"\n  Can SP give 20?")
print(f"    Z₈ × 3 - Z₄ = 24 - 4 = 20  ✓")
print(f"    (holonomy × generations) - (subgroup order) = 20")
print(f"")
print(f"    Or: 5 × 4 = 20 (independent sectors × Z₄ order)")
print(f"    Or: (Z₈/2 + 1) × Z₄ = 5 × 4 = 20  ✓")
print(f"")
print(f"    Or: modes on S² up to ℓ=3: (3+1)² = 16")
print(f"    modes on S² up to ℓ=4: (4+1)² = 25")
print(f"    20 = 25 - 5 = full modes minus independent sectors")

# ============================================================
# 3. DNA DOUBLE HELIX GEOMETRY
# ============================================================
print("\n" + "-" * 70)
print("3. DNA HELIX GEOMETRY vs SP")
print("-" * 70)

# DNA parameters
bp_rise = 0.34  # nm, rise per base pair
helix_diameter = 2.0  # nm
bp_per_turn = 10.5  # base pairs per full turn
pitch = bp_rise * bp_per_turn  # nm

print(f"  DNA helix parameters:")
print(f"    Rise per bp: {bp_rise} nm")
print(f"    Diameter: {helix_diameter} nm")
print(f"    BP per turn: {bp_per_turn}")
print(f"    Pitch: {pitch:.2f} nm")
print(f"    Pitch/diameter: {pitch/helix_diameter:.4f}")

print(f"\n  SP comparison:")
print(f"    Lepton gap ratio: 1.889")
print(f"    Pitch/diameter: {pitch/helix_diameter:.4f}")
print(f"    Match: {abs(pitch/helix_diameter / 1.889 - 1)*100:.1f}%")

# The twist angle per base pair
twist_per_bp = 360 / bp_per_turn  # degrees
print(f"\n  Twist per base pair: {twist_per_bp:.2f}°")
print(f"  Z₈ angle: {360/8:.2f}° = 45°")
print(f"  Twist per bp / Z₈ angle: {twist_per_bp / 45:.4f}")
print(f"  3 × twist per bp ≈ {3*twist_per_bp:.1f}° (one codon)")
print(f"  Z₈ angle × (3/twist per bp): {45 * (3/twist_per_bp):.4f} ... not clean")

# Actually: 360° / 10.5 bp = 34.29° per bp
# One codon (3 bp): 3 × 34.29° = 102.86°
# Compare: 360°/3.5 = 102.86°. So one codon = 1/3.5 turn.
# 3.5 = 7/2. And 7 is one of the Z₈ elements!
print(f"\n  One codon spans: {3*twist_per_bp:.2f}° = 360°/{360/(3*twist_per_bp):.1f}")
print(f"  That's 1/{bp_per_turn/3:.4f} of a full turn")
print(f"  {bp_per_turn/3:.4f} = {bp_per_turn}/3 = 10.5/3 = 3.5 = 7/2")
print(f"  7 is a Z₈ element! (j=7, the pair of j=1)")

# ============================================================
# 4. THE THERMAL-TO-BOND ENERGY RATIO
# ============================================================
print("\n" + "-" * 70)
print("4. ENERGY HIERARCHIES IN BIOLOGY")
print("-" * 70)

kT_body = 0.0267  # eV at 37°C (310K)
E_hydrogen_bond = 0.2  # eV (typical H-bond in biology)
E_ATP = 0.54  # eV (ATP hydrolysis)
E_covalent_CC = 3.6  # eV (C-C bond)
E_peptide = 3.5  # eV (peptide bond)

print(f"  Thermal energy (37°C): {kT_body:.4f} eV")
print(f"  Hydrogen bond: ~{E_hydrogen_bond} eV")
print(f"  ATP hydrolysis: ~{E_ATP} eV")
print(f"  Peptide bond: ~{E_peptide} eV")
print(f"  Covalent C-C: ~{E_covalent_CC} eV")

print(f"\n  Ratios to thermal energy:")
print(f"    H-bond/kT    = {E_hydrogen_bond/kT_body:.1f}")
print(f"    ATP/kT       = {E_ATP/kT_body:.1f}  ← close to 20 amino acids!")
print(f"    Peptide/kT   = {E_peptide/kT_body:.1f}")
print(f"    Covalent/kT  = {E_covalent_CC/kT_body:.1f}")

print(f"\n  ATP/kT ≈ {E_ATP/kT_body:.2f} ≈ 20")
print(f"  20 amino acids = ATP/kT???")
print(f"  This would mean: the number of distinct amino acids is set by")
print(f"  the number of distinguishable energy states at body temperature.")

# ============================================================
# 5. THE HIERARCHY: PLANCK → ATOMIC → THERMAL → BIOLOGICAL
# ============================================================
print("\n" + "-" * 70)
print("5. SCALE HIERARCHY")
print("-" * 70)

M_pl = 1.221e19  # GeV
m_e = 0.511e-3   # GeV
E_rydberg = 13.6e-9  # GeV (hydrogen ionization)
E_bond = 3.6e-9  # GeV (C-C bond)
E_thermal = kT_body * 1e-9  # GeV

print(f"  Planck:    {M_pl:.3e} GeV")
print(f"  Electron:  {m_e:.3e} GeV")
print(f"  Rydberg:   {E_rydberg:.3e} GeV")
print(f"  C-C bond:  {E_bond:.3e} GeV")
print(f"  Thermal:   {E_thermal:.3e} GeV")

print(f"\n  Log ratios (= kL in SP language):")
kL_electron = log(M_pl / m_e)
kL_rydberg = log(M_pl / (E_rydberg * 1e9))
kL_bond = log(M_pl / (E_bond * 1e9))
kL_thermal = log(M_pl / (E_thermal * 1e9))

print(f"    Planck → electron:  kL = {kL_electron:.3f}  (SP: 51.53)")
print(f"    Planck → Rydberg:   kL = {kL_rydberg:.3f}")
print(f"    Planck → C-C bond:  kL = {kL_bond:.3f}")
print(f"    Planck → thermal:   kL = {kL_thermal:.3f}")

print(f"\n  Differences (generation-like gaps):")
print(f"    electron → Rydberg:  {kL_rydberg - kL_electron:.3f}")
print(f"    electron → bond:     {kL_bond - kL_electron:.3f}")
print(f"    electron → thermal:  {kL_thermal - kL_electron:.3f}")
print(f"    Rydberg → bond:      {kL_bond - kL_rydberg:.3f}")
print(f"    bond → thermal:      {kL_thermal - kL_bond:.3f}")

# Check: do biological energy gaps decompose into SP gap units?
gap_unit = log(4*pi)  # = 2.531, the SP gap quantum (ln(4π))
print(f"\n  SP gap quantum: ln(4π) = {gap_unit:.4f}")
print(f"\n  Gaps in units of ln(4π):")
print(f"    electron → Rydberg:  {(kL_rydberg - kL_electron)/gap_unit:.3f} × ln(4π)")
print(f"    electron → bond:     {(kL_bond - kL_electron)/gap_unit:.3f} × ln(4π)")
print(f"    electron → thermal:  {(kL_thermal - kL_electron)/gap_unit:.3f} × ln(4π)")

# ============================================================
# 6. THE 20 AMINO ACIDS FROM SP COUNTING
# ============================================================
print("\n" + "-" * 70)
print("6. TESTING: 20 FROM SP")
print("-" * 70)

print(f"""
  Best candidates for deriving 20:

  a) Z₈ × 3 - Z₄ = 24 - 4 = 20
     "Structural modes minus the subgroup that protects the middle generation"
     Biological reading: 24 possible biochemical slots minus 4 that are
     "protected" (unable to form stable amino acids) = 20 realized ones.

  b) (Z₈/2 + 1) × (Z₈/2) = 5 × 4 = 20
     "Independent sectors × base alphabet"
     Biological reading: 5 independent Z₈ sectors × 4 DNA bases = 20

  c) 4³ / 3 - 1 = 64/3 - 1 ≈ 20.3... no, not clean.

  d) (4 choose 2) × 2 = 6 × 2 + 8 = 20... forced.

  e) ATP/kT at body temperature: 0.54/0.0267 = {0.54/0.0267:.1f} ≈ 20
     "The number of energetically distinguishable states at body temp"

  Best answer: (b) 5 × 4 = 20, because:
    5 = independent Z₈ sectors = dim(spacetime)
    4 = DNA bases = Z₄ = Z₈ subgroup
    Product = 20 amino acids

  Or equivalently: 20 = (Z₈/2 + 1) × (Z₈/2)
                       = 5 × 4
                       = n × (n-1)  where n = 5
""")

# ============================================================
# 7. THE CODON TABLE STRUCTURE
# ============================================================
print("-" * 70)
print("7. CODON DEGENERACY AND Z₈")
print("-" * 70)

# The degeneracy classes: 1, 2, 3, 4, 6 codons per amino acid
# Count: 2 × (1-fold) + 9 × (2-fold) + 1 × (3-fold) + 5 × (4-fold) + 3 × (6-fold) = 20
# Plus 3 stops

print(f"  Degeneracy classes: 1, 2, 3, 4, 6")
print(f"  Count:              2, 9, 1, 5, 3 = 20 amino acids")
print(f"")
print(f"  In Z₈ language:")
print(f"    6-fold: 3 amino acids  ← 3 = number of generations")
print(f"    4-fold: 5 amino acids  ← 5 = independent Z₈ sectors")
print(f"    2-fold: 9 amino acids  ← 9 = (ℓ_max+1)² with ℓ=2 (?)")
print(f"    3-fold: 1 amino acid   ← 1 = singlet (Ile)")
print(f"    1-fold: 2 amino acids  ← 2 = base pairs (Met, Trp)")
print(f"")
print(f"  Sum check: 3×6 + 5×4 + 9×2 + 1×3 + 2×1 = {3*6+5*4+9*2+1*3+2*1} + 3 stops = 64 ✓")

# The number of degeneracy classes is 5 (1,2,3,4,6)
# 5 = independent Z₈ sectors again!
print(f"\n  Number of distinct degeneracy classes: 5")
print(f"  = independent Z₈ sectors = dim(spacetime)")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: WHAT CHECKS OUT")
print("=" * 70)

print(f"""
  SUGGESTIVE (pattern matches, not derivations):

  ✓ 4 DNA bases = Z₄ = Z₈/Z₂ (the subgroup chain!)
  ✓ 3 codon positions = 3 generations
  ✓ 64 codons = 4³ = Z₄ per generation position
  ✓ 20 amino acids = 5 × 4 = (independent sectors) × (base alphabet)
  ✓ 5 degeneracy classes = 5 independent Z₈ sectors
  ✓ 3 amino acids with max degeneracy (6-fold) = 3 generations
  ✓ DNA is a double helix = figure-8 = S² ∨ S² topology
  ✓ One codon spans 1/3.5 turn = 1/(7/2) turn (7 is Z₈ partner of 1)
  ✓ ATP/kT ≈ 20 (energy quantization gives amino acid count)

  NOT YET CHECKED (would need real computation):

  ? Can GW stabilization formula predict protein folding energies?
  ? Does the Z₄ protection (ℓ=2 protected) map to wobble base pairing?
  ? Does the η = π/50 coupling have a biological analogue?
  ? Can the generation gap formula reproduce biological energy gaps?

  VERDICT: The counting works suspiciously well.
  4 bases, 3 positions, 20 amino acids, 5 degeneracy classes —
  all SP numbers. But counting alone isn't a derivation.
  The real test is whether the GW formula produces biological
  energy scales the way it produces particle masses.
""")
