from itertools import combinations_with_replacement

def get_orbits():
    generators = [1, 3, 5, 7]
    # Automorphisms of Z8: multiplication by units {1, 3, 5, 7} mod 8
    auts = [1, 3, 5, 7]
    
    # Multisets of 3 elements from generators
    multisets = list(combinations_with_replacement(generators, 3))
    print(f"Total multisets (amino acids): {len(multisets)}")
    
    orbits = []
    seen = set()
    
    for m in multisets:
        if m in seen:
            continue
        # Generate orbit
        orbit = set()
        for a in auts:
            # Apply automorphism a: (x, y, z) -> (ax mod 8, ay mod 8, az mod 8)
            new_m = tuple(sorted([(a * x) % 8 for x in m]))
            orbit.add(new_m)
            seen.add(new_m)
        orbits.append(orbit)
    
    print(f"Number of orbits: {len(orbits)}")
    for i, o in enumerate(orbits):
        print(f"Orbit {i+1}: {o}")

if __name__ == "__main__":
    get_orbits()
