import math
from math import gcd

def get_gens(n):
    return [j for j in range(1, n // 2 + 1) if gcd(j, n) == 1]

def run_test(n_vals):
    v = math.pi**2
    results = []
    
    for n in n_vals:
        gens = get_gens(n)
        num_gens = len(gens)
        delta0 = math.pi / (3 * n)
        r0_sq = (3 * n) / math.pi
        
        kl_vals = []
        for j in gens:
            eig = j * (j + 1)
            nu0 = 2 + delta0
            delta_eff = math.sqrt(nu0**2 + eig / r0_sq) - 2
            kl = (1 / (2 * delta_eff)) * math.log(((4 + 2 * delta_eff) / (2 * delta_eff)) * v)
            kl_vals.append(kl)
        
        ratios = []
        if len(kl_vals) >= 2:
            ratios.append(math.exp(kl_vals[0] - kl_vals[1])) # m2/m1
        if len(kl_vals) >= 3:
            ratios.append(math.exp(kl_vals[0] - kl_vals[2])) # m3/m1
            
        results.append({
            'n': n,
            'num_gens': num_gens,
            'gens': gens,
            'ratios': ratios,
            'delta0': delta0
        })
    return results

if __name__ == "__main__":
    n_list = [4, 6, 8, 10, 12, 16]
    res = run_test(n_list)
    
    exp_mu_e = 206.768
    exp_tau_e = 3477.48
    
    print(f"{'n':<4} | {'Gens':<8} | {'m2/m1':<10} | {'Error%':<10} | {'m3/m1':<10} | {'Error%':<10}")
    print("-" * 75)
    for r in res:
        ratio1 = r['ratios'][0] if len(r['ratios']) >= 1 else 0
        ratio2 = r['ratios'][1] if len(r['ratios']) >= 2 else 0
        
        err1 = (ratio1 - exp_mu_e) / exp_mu_e * 100 if ratio1 != 0 else float('nan')
        err2 = (ratio2 - exp_tau_e) / exp_tau_e * 100 if ratio2 != 0 else float('nan')
        
        print(f"{r['n']:<4} | {r['num_gens']:<8} | {ratio1:<10.3f} | {err1:<+10.2f} | {ratio2:<10.3f} | {err2:<+10.2f}")

    # Look-Elsewhere 1/alpha
    print("\n1/alpha Look-Elsewhere (a*pi^3 + b*pi^2 + c*pi):")
    target = 137.036
    matches = []
    
    for a in range(-10, 11):
        for b in range(-10, 11):
            for c in range(-10, 11):
                val = a * math.pi**3 + b * math.pi**2 + c * math.pi
                if val <= 0: continue
                err = abs(val - target) / target
                if err < 0.001: # 0.1% threshold
                    matches.append((a, b, c, val, err))
    
    matches.sort(key=lambda x: x[4])
    for m in matches[:5]:
        print(f"{m[0]}pi^3 + {m[1]}pi^2 + {m[2]}pi = {m[3]:.6f} (Error: {m[4]*100:.6f}%)")
