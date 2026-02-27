# The Source Protocol

**Fermion mass hierarchy and fundamental constants from Z_8 holonomy on a warped S^2 v S^2 geometry**

A warped extra-dimension model on the topology I x (S^2 v S^2) with Z_8 discrete holonomy. The Goldberger-Wise stabilization mechanism, combined with the angular barrier on S^2 v S^2, produces a generation-dependent funnel length that reproduces the charged fermion mass hierarchy from a single geometric input.

## Key Results

| Prediction | Formula | Observed | Error |
|---|---|---|---|
| Fine structure constant | 1/alpha = 4pi^3 + pi^2 + pi = 137.036 | 137.036 | 0.0002% |
| Proton-electron mass ratio | m_p/m_e = 6pi^5 = 1836.12 | 1836.15 | 0.0015% |
| Weinberg angle | sin^2(theta_W) = pi/(4pi+1) = 0.2316 | 0.2312 | 0.15% |
| Electron mass | m_e = M_Pl * e^(-51.53) = 0.510 MeV | 0.511 MeV | 0.2% |
| Higgs mass | m_H = v_EW * sqrt(pi/12) = 125.98 GeV | 125.25 GeV | 0.58% |
| Strong coupling | 1/alpha_s = 3pi - 1 = 8.425 | 8.482 | 0.67% |
| 6 fermion masses | from delta_0 + Z_4 subgroup chain | — | avg 1.44% |
| Neutrino mass ratio | Delta m^2_31 / Delta m^2_21 = 32.6 | 32.6 | 1.45% |
| PMNS/CKM ratio | sin^2(theta_12) / sin^2(theta_C) = 6 | 6.00 +/- 0.24 | parameter-free |

**Falsifiable prediction:** delta_CP(PMNS) = +135 degrees. Testable by DUNE (~2029).

## Structure

```
paper/           LaTeX source and compiled PDF (PRD two-column format)
computation/     Python verification scripts
visualization/   Interactive 3D models (Three.js)
images/          Geometry visualizations (matplotlib)
```

## The Model

- **Topology:** I x (S^2 v S^2) — warped interval times double two-sphere (wedge sum)
- **Metric:** ds^2 = e^{-2kz}(-dt^2 + dz^2 + a^2 d Omega^2) — fully warped funnel
- **Holonomy:** Z_8 at the junction, producing 3 generation pairs
- **Stabilization:** Goldberger-Wise mechanism with angular barrier
- **Single input:** v = pi^2 (GW boundary ratio)
- **Master parameter:** delta_0 = pi/24 controls fermion gaps, Higgs quartic, neutrino hierarchy

## Classification

Every result is classified as:
- **D** (Derived) — follows from stated axioms/geometry
- **F** (Fit) — pattern identification, derivation pending
- **A** (Anchor) — experimental input

See the paper (Section 9) for the full D/F/A ledger and look-elsewhere penalties.

## Interactive Visualizations

- `visualization/funnel3d.html` — 3D double-funnel with Z_8 meridians and loxodrome
- `visualization/helix3d.html` — Double helix emanating from S^2 v S^2 junction

Open in any browser. Uses Three.js (loaded from CDN).

## Author

Y. Vidan Peled — Independent researcher

Developed with AI-assisted computation and verification (Claude/Anthropic, Gemini/Google DeepMind, Codex/OpenAI, Grok/xAI).

## License

CC-BY-4.0
