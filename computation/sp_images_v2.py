#!/usr/bin/env python3
"""
Source Protocol — Geometry Visualizations v2
Emphasis on curvature compression approaching the singularity.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# IMAGE 1v2: Double Funnel — wireframe emphasis, visible warping
# ============================================================
def render_funnel_wireframe():
    fig = plt.figure(figsize=(12, 16), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    k = 0.9
    z_max = 4.0

    # Ring lines at constant z — NON-UNIFORM spacing (dense near throat)
    # This IS the curvature — the compression of the grid IS the warping
    z_pos = np.exp(np.linspace(np.log(0.03), np.log(z_max), 50))
    z_rings = np.sort(np.concatenate([-z_pos, [0], z_pos]))

    for z_val in z_rings:
        r = np.exp(-k * abs(z_val))
        t = np.linspace(0, 2*np.pi, 200)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.full_like(t, z_val)

        # Brightness and thickness increase near throat
        proximity = r  # r→1 at throat, r→0 far away... wait, r→1 far from throat
        # Actually r = e^{-k|z|}, so r→1 near throat (z=0), r→small far away
        intensity = r ** 0.5
        ax.plot(x, y, z, color=(0.6*intensity + 0.2, 0.3*intensity + 0.1, intensity, 0.3 + 0.6*intensity),
                linewidth=0.3 + 2.5*intensity)

    # Meridian lines — 16 of them
    for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
        z_line = np.linspace(-z_max, z_max, 800)
        r_line = np.exp(-k * np.abs(z_line))
        x_line = r_line * np.cos(angle)
        y_line = r_line * np.sin(angle)

        # Color by proximity to throat
        for i in range(len(z_line)-1):
            intensity = r_line[i] ** 0.5
            ax.plot(x_line[i:i+2], y_line[i:i+2], z_line[i:i+2],
                    color=(0.5*intensity+0.2, 0.2*intensity+0.1, intensity, 0.15 + 0.4*intensity),
                    linewidth=0.2 + 1.0*intensity)

    # Bright throat ring
    t = np.linspace(0, 2*np.pi, 300)
    r_throat = np.exp(-k * 0.03)
    ax.plot(r_throat*np.cos(t), r_throat*np.sin(t), np.full_like(t, 0),
            color='white', linewidth=2.0, alpha=0.8)

    # Throat point
    ax.scatter([0], [0], [0], color='white', s=50, zorder=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-z_max, z_max)
    ax.set_axis_off()
    ax.view_init(elev=10, azim=30)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_01v2_funnel_wireframe.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 1v2: Funnel wireframe saved")


# ============================================================
# IMAGE 2v2: Throat extreme close-up — the singularity
# ============================================================
def render_singularity():
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    k = 2.0  # steep warp for close-up drama

    # Very dense ring grid near z=0
    z_pos = np.concatenate([
        np.linspace(0.005, 0.05, 20),   # extremely close to throat
        np.linspace(0.06, 0.2, 15),
        np.linspace(0.25, 0.8, 12),
    ])
    z_rings = np.sort(np.concatenate([-z_pos, z_pos]))

    for z_val in z_rings:
        r = np.exp(-k * abs(z_val))
        t = np.linspace(0, 2*np.pi, 300)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.full_like(t, z_val)

        intensity = r ** 0.3
        # White-blue near throat, fading to deep blue far away
        ax.plot(x, y, z,
                color=(0.5 + 0.5*intensity, 0.7 + 0.3*intensity, 1.0, 0.2 + 0.7*intensity),
                linewidth=0.2 + 3.0*intensity)

    # 8 meridians — Z₈
    for i in range(8):
        angle = i * np.pi / 4
        z_line = np.linspace(-0.8, 0.8, 400)
        r_line = np.exp(-k * np.abs(z_line))
        x_line = r_line * np.cos(angle)
        y_line = r_line * np.sin(angle)
        ax.plot(x_line, y_line, z_line, color=(0.6, 0.4, 1.0, 0.4), linewidth=0.8)

    # The singularity — glowing point
    ax.scatter([0], [0], [0], color='white', s=100, zorder=10)
    # Glow rings
    for rad in np.linspace(0.005, 0.04, 8):
        phi = np.linspace(0, 2*np.pi, 100)
        ax.plot(rad*np.cos(phi), rad*np.sin(phi), np.zeros(100),
                color='white', alpha=0.4, linewidth=1.5)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.8, 0.8)
    ax.set_axis_off()
    ax.view_init(elev=8, azim=25)
    ax.dist = 6.5

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_02v2_singularity.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 2v2: Singularity close-up saved")


# ============================================================
# IMAGE 3v2: Loxodrome — glowing helix on visible sphere
# ============================================================
def render_loxodrome_v2():
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Visible sphere grid
    u = np.linspace(0, 2*np.pi, 200)
    v_ang = np.linspace(0, np.pi, 100)

    # Latitude lines
    for lat in np.linspace(0.1, np.pi-0.1, 15):
        x_lat = np.cos(u) * np.sin(lat)
        y_lat = np.sin(u) * np.sin(lat)
        z_lat = np.full_like(u, np.cos(lat))
        ax.plot(x_lat, y_lat, z_lat, color='#252560', linewidth=0.5, alpha=0.5)

    # Longitude lines
    for lon in np.linspace(0, 2*np.pi, 24, endpoint=False):
        x_lon = np.cos(lon) * np.sin(v_ang)
        y_lon = np.sin(lon) * np.sin(v_ang)
        z_lon = np.cos(v_ang)
        ax.plot(x_lon, y_lon, z_lon, color='#252560', linewidth=0.5, alpha=0.5)

    # The loxodrome — alpha = pi/4 (45 degree crossing angle)
    # This gives length L = pi*sqrt(2) on unit sphere
    t = np.linspace(-6, 6, 5000)
    phi = 2 * np.arctan(np.exp(-t))  # cot(pi/4) = 1

    x_lox = np.cos(t) * np.sin(phi)
    y_lox = np.sin(t) * np.sin(phi)
    z_lox = np.cos(phi)

    # Draw with glow effect — multiple passes at decreasing alpha and increasing width
    glow_colors = [
        (3.0, 0.08, '#ffffff'),
        (2.2, 0.15, '#aaccff'),
        (1.5, 0.25, '#6688ff'),
        (1.0, 0.9, '#ff6600'),
    ]
    for width, alpha, color in glow_colors:
        ax.plot(x_lox, y_lox, z_lox, color=color, linewidth=width, alpha=alpha)

    # Gradient color version on top
    n = len(t)
    colors = plt.cm.inferno(np.linspace(0.2, 0.9, n))
    for i in range(0, n-1, 3):
        ax.plot(x_lox[i:i+4], y_lox[i:i+4], z_lox[i:i+4],
                color=colors[i], linewidth=1.8, alpha=0.85)

    # Poles — junction points
    ax.scatter([0, 0], [0, 0], [1, -1], color='white', s=120, zorder=10)
    # Pole glow
    for pole_z in [1, -1]:
        for rad in np.linspace(0.02, 0.1, 6):
            phi_g = np.linspace(0, 2*np.pi, 50)
            ax.plot(rad*np.cos(phi_g), rad*np.sin(phi_g),
                    np.full(50, pole_z), color='white', alpha=0.2, linewidth=0.8)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=40)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_03v2_loxodrome.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 3v2: Loxodrome saved")


# ============================================================
# IMAGE 7: The full picture — funnel cross-section with
# loxodrome helix spiraling down the throat
# ============================================================
def render_funnel_with_helix():
    fig = plt.figure(figsize=(12, 18), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    k = 0.7
    z_max = 5.0

    # Funnel surface — semi-transparent
    z = np.linspace(-z_max, z_max, 300)
    theta = np.linspace(0, 2*np.pi, 150)
    Z, Theta = np.meshgrid(z, theta)
    R = np.exp(-k * np.abs(Z))
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Color by curvature
    curv = 1.0 / (R + 0.02)
    curv_norm = curv / curv.max()

    cmap_funnel = LinearSegmentedColormap.from_list('funnel', [
        (0.0, '#050515'), (0.3, '#100830'), (0.6, '#201060'),
        (0.8, '#4020a0'), (0.95, '#8040e0'), (1.0, '#c080ff'),
    ])

    ax.plot_surface(X, Y, Z, facecolors=cmap_funnel(curv_norm),
                    alpha=0.25, rstride=3, cstride=3, shade=False)

    # Grid rings
    z_pos = np.exp(np.linspace(np.log(0.05), np.log(z_max), 35))
    z_rings = np.sort(np.concatenate([-z_pos, z_pos]))
    for z_val in z_rings:
        r = np.exp(-k * abs(z_val))
        t = np.linspace(0, 2*np.pi, 200)
        intensity = r ** 0.5
        ax.plot(r*np.cos(t), r*np.sin(t), np.full_like(t, z_val),
                color=(0.4+0.3*intensity, 0.2+0.2*intensity, 0.7+0.3*intensity,
                       0.1 + 0.3*intensity),
                linewidth=0.2 + 1.0*intensity)

    # THE HELIX — a loxodrome spiraling down through the funnel
    # This is the key image: the helix IS inside the funnel
    t_helix = np.linspace(-z_max, z_max, 3000)
    r_helix = np.exp(-k * np.abs(t_helix)) * 0.85  # slightly inside the surface
    # Spiral with increasing frequency near the throat
    n_turns = 12
    phase = n_turns * 2 * np.pi * np.sign(t_helix) * (1 - np.exp(-2*np.abs(t_helix)))
    x_helix = r_helix * np.cos(phase)
    y_helix = r_helix * np.sin(phase)

    # Glow
    ax.plot(x_helix, y_helix, t_helix, color='#ff8800', linewidth=0.5, alpha=0.15)
    ax.plot(x_helix, y_helix, t_helix, color='#ffaa33', linewidth=1.5, alpha=0.5)
    ax.plot(x_helix, y_helix, t_helix, color='#ffcc66', linewidth=0.8, alpha=0.8)

    # Throat
    ax.scatter([0], [0], [0], color='white', s=60, zorder=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-z_max, z_max)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=35)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_07_funnel_helix.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 7: Funnel with helix saved")


# ============================================================
# IMAGE 8: Curvature intensity — 2D heatmap, portrait format
# The singularity as pure curvature concentration
# ============================================================
def render_curvature_intensity():
    fig, ax = plt.subplots(figsize=(10, 18), facecolor='black')
    ax.set_facecolor('black')

    k = 0.8
    # Create a 2D field where x = radial, y = z (extra dimension)
    z = np.linspace(-5, 5, 1000)
    x = np.linspace(-1.5, 1.5, 600)
    Z, X = np.meshgrid(z, x)

    # Funnel boundary
    R_bound = np.exp(-k * np.abs(Z))
    inside = np.abs(X) < R_bound

    # Ricci scalar of RS: R ~ -20k² (constant in bulk, diverges at branes)
    # But visually: the "energy density" of curvature ~ e^{2k|z|} * 1/(r²)
    # This captures both the warp factor growth and the radial focusing
    curvature = np.zeros_like(Z)
    curvature[inside] = np.exp(1.6 * np.abs(Z[inside])) / (np.abs(X[inside]) + 0.005)

    # Normalize
    pct = np.percentile(curvature[inside], 98)
    curvature_norm = np.clip(curvature / pct, 0, 1)
    curvature_norm[~inside] = 0

    cmap_heat = LinearSegmentedColormap.from_list('heat', [
        (0.0, '#000005'),
        (0.15, '#0a0030'),
        (0.3, '#200060'),
        (0.45, '#4000a0'),
        (0.6, '#8000c0'),
        (0.75, '#d040a0'),
        (0.85, '#ff6040'),
        (0.92, '#ffcc20'),
        (1.0, '#ffffff'),
    ])

    ax.imshow(curvature_norm, extent=[-1.5, 1.5, -5, 5],
              aspect='auto', cmap=cmap_heat, alpha=0.95, origin='lower',
              interpolation='bilinear')

    # Funnel outline
    ax.plot(np.exp(-k*np.abs(z)), z, color='#ffffff', linewidth=0.8, alpha=0.3)
    ax.plot(-np.exp(-k*np.abs(z)), z, color='#ffffff', linewidth=0.8, alpha=0.3)

    # Horizontal grid — non-uniform, dense at throat
    z_pos = np.exp(np.linspace(np.log(0.02), np.log(5), 40))
    z_grid = np.sort(np.concatenate([-z_pos, z_pos]))
    for z_val in z_grid:
        r = np.exp(-k * abs(z_val))
        intensity = r ** 0.5
        ax.plot([-r, r], [z_val, z_val],
                color='white', linewidth=0.15 + 0.6*intensity,
                alpha=0.05 + 0.2*intensity)

    # The singularity point
    ax.plot(0, 0, 'o', color='white', markersize=4, zorder=10)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-5, 5)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_08_curvature_heat.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("Image 8: Curvature intensity saved")


# ============================================================
# IMAGE 9: The complete geometry — bird's eye looking DOWN
# into the throat. Concentric rings compressing to a point.
# ============================================================
def render_looking_down():
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    k = 0.8

    # Concentric circles at different z values
    # The radius = e^{-k*z}, so circles get smaller as z increases (deeper into throat)
    z_values = np.concatenate([
        np.linspace(0, 0.3, 15),    # very close, tightly packed
        np.linspace(0.35, 1.0, 12),
        np.linspace(1.1, 2.5, 10),
        np.linspace(2.8, 5.0, 8),
    ])

    theta = np.linspace(0, 2*np.pi, 500)

    for z_val in z_values:
        r = np.exp(-k * z_val)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Deeper = brighter (more curvature concentrated)
        depth_frac = z_val / 5.0
        # Inner rings: hot. Outer rings: cool.
        if depth_frac < 0.3:
            color = (1.0, 0.8 + 0.2*(1-depth_frac/0.3), 1.0, 0.4 + 0.5*(1-depth_frac/0.3))
        elif depth_frac < 0.6:
            f = (depth_frac - 0.3) / 0.3
            color = (0.4 + 0.6*(1-f), 0.2 + 0.2*(1-f), 0.8 + 0.2*(1-f), 0.15 + 0.25*(1-f))
        else:
            f = (depth_frac - 0.6) / 0.4
            color = (0.15 + 0.1*(1-f), 0.1 + 0.05*(1-f), 0.3 + 0.2*(1-f), 0.08 + 0.1*(1-f))

        lw = 0.3 + 2.5 * (1 - depth_frac)
        ax.plot(x, y, color=color, linewidth=lw)

    # 8 radial lines — Z₈ symmetry
    for i in range(8):
        angle = i * np.pi / 4
        r_max = np.exp(0)  # = 1
        r_min = np.exp(-k * 5.0)
        rs = np.linspace(r_min, r_max, 200)
        xs = rs * np.cos(angle)
        ys = rs * np.sin(angle)
        # Gradient along each radial line
        for j in range(len(rs)-1):
            frac = 1 - rs[j]  # 0 at outer, ~1 at inner
            ax.plot(xs[j:j+2], ys[j:j+2],
                    color=(0.3+0.5*frac, 0.2+0.3*frac, 0.6+0.4*frac, 0.05+0.3*frac),
                    linewidth=0.3 + 1.5*frac)

    # The loxodrome projected from above — it appears as a spiral
    t = np.linspace(0.01, 5, 2000)
    r_lox = np.exp(-k * t)
    phase = 8 * t  # winding
    x_lox = r_lox * np.cos(phase)
    y_lox = r_lox * np.sin(phase)

    # Glow effect
    ax.plot(x_lox, y_lox, color='#ff6600', linewidth=0.3, alpha=0.1)
    ax.plot(x_lox, y_lox, color='#ffaa44', linewidth=1.2, alpha=0.4)
    ax.plot(x_lox, y_lox, color='#ffcc88', linewidth=0.6, alpha=0.7)

    # Center point — the singularity
    ax.plot(0, 0, 'o', color='white', markersize=6, zorder=10)
    # Glow
    for rad in np.linspace(0.003, 0.03, 8):
        circle = plt.Circle((0, 0), rad, fill=False, color='white',
                           alpha=0.3, linewidth=1.0)
        ax.add_patch(circle)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_09_looking_down.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("Image 9: Looking down the throat saved")


if __name__ == '__main__':
    render_funnel_wireframe()
    render_singularity()
    render_loxodrome_v2()
    render_funnel_with_helix()
    render_curvature_intensity()
    render_looking_down()
    print("\nAll v2 images saved to /home/ai-guest/sp-computation/")
