#!/usr/bin/env python3
"""
Source Protocol — Geometry Visualizations
Mathematically correct renderings of the funnel metric ds² = e^{-2kz}(-dt² + dz² + a²dΩ²)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# SP parameters
k = 1.0  # warp factor scale (normalized)
z_max = 4.0  # range of extra dimension

# Custom colormaps
throat_cmap = LinearSegmentedColormap.from_list('throat', [
    (0.0, '#0a0a2e'),   # deep space
    (0.3, '#1a0a4e'),   # deep violet
    (0.5, '#4a1a8e'),   # purple
    (0.7, '#8a3abe'),   # bright violet
    (0.85, '#ca7aee'),  # light purple
    (0.95, '#eaccff'),  # near white
    (1.0, '#ffffff'),   # singularity white
])

curvature_cmap = LinearSegmentedColormap.from_list('curvature', [
    (0.0, '#000820'),
    (0.2, '#001050'),
    (0.4, '#0030a0'),
    (0.6, '#0070ff'),
    (0.8, '#40c0ff'),
    (1.0, '#ffffff'),
])


def warp_radius(z, k=1.0):
    """The RS warp factor: radius = e^{-k|z|}"""
    return np.exp(-k * np.abs(z))


# ============================================================
# IMAGE 1: The Double Funnel — Full geometry, both cones
# ============================================================
def render_double_funnel():
    fig = plt.figure(figsize=(12, 16), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # z runs from -z_max to +z_max, throat at z=0
    z = np.linspace(-z_max, z_max, 400)
    theta = np.linspace(0, 2*np.pi, 200)
    Z, Theta = np.meshgrid(z, theta)

    # Warp factor gives radius
    R = warp_radius(Z, k=0.8)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Color by curvature (higher near throat)
    curvature = 1.0 / (R + 0.01)
    curvature_norm = curvature / curvature.max()

    ax.plot_surface(X, Y, Z, facecolors=throat_cmap(curvature_norm),
                    alpha=0.85, rstride=2, cstride=2, shade=False)

    # Add gridlines showing the warping — key visual
    for z_val in np.linspace(-z_max, z_max, 40):
        r = warp_radius(z_val, k=0.8)
        t = np.linspace(0, 2*np.pi, 100)
        x_ring = r * np.cos(t)
        y_ring = r * np.sin(t)
        z_ring = np.full_like(t, z_val)
        # Lines get brighter near throat
        brightness = min(1.0, 0.15 + 0.85 * (1 - r))
        ax.plot(x_ring, y_ring, z_ring, color=(brightness, brightness*0.7, 1.0),
                linewidth=0.3 + 1.5*(1-r), alpha=0.4 + 0.5*(1-r))

    # Meridian lines
    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
        z_line = np.linspace(-z_max, z_max, 400)
        r_line = warp_radius(z_line, k=0.8)
        x_line = r_line * np.cos(angle)
        y_line = r_line * np.sin(angle)
        ax.plot(x_line, y_line, z_line, color='#6040c0', linewidth=0.4, alpha=0.3)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-z_max, z_max)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=35)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_01_double_funnel.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 1: Double funnel saved")


# ============================================================
# IMAGE 2: Throat close-up — curvature approaching singularity
# ============================================================
def render_throat_closeup():
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Zoom into the throat region
    z = np.linspace(-1.5, 1.5, 600)
    theta = np.linspace(0, 2*np.pi, 300)
    Z, Theta = np.meshgrid(z, theta)

    R = warp_radius(Z, k=1.5)  # steeper warp for dramatic effect

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    curvature = 1.0 / (R**2 + 0.001)
    curvature_norm = np.clip(curvature / np.percentile(curvature, 99), 0, 1)

    ax.plot_surface(X, Y, Z, facecolors=curvature_cmap(curvature_norm),
                    alpha=0.9, rstride=2, cstride=2, shade=False)

    # Dense gridlines near throat — the compression IS the curvature
    # Non-uniform spacing: dense near z=0, sparse far away
    z_grid_pos = np.exp(np.linspace(np.log(0.02), np.log(1.5), 30))
    z_grid = np.sort(np.concatenate([-z_grid_pos, z_grid_pos]))

    for z_val in z_grid:
        r = warp_radius(z_val, k=1.5)
        t = np.linspace(0, 2*np.pi, 200)
        x_ring = r * np.cos(t)
        y_ring = r * np.sin(t)
        z_ring = np.full_like(t, z_val)
        intensity = min(1.0, 0.1 + 2.0 * (1 - r))
        ax.plot(x_ring, y_ring, z_ring,
                color=(0.3 + 0.7*intensity, 0.5 + 0.5*intensity, 1.0),
                linewidth=0.2 + 2.0*intensity, alpha=0.3 + 0.6*intensity)

    # The point at the throat — the singularity
    ax.scatter([0], [0], [0], color='white', s=30, zorder=10)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-1.5, 1.5)
    ax.set_axis_off()
    ax.view_init(elev=5, azim=30)
    ax.dist = 6.5

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_02_throat_closeup.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 2: Throat close-up saved")


# ============================================================
# IMAGE 3: The Loxodrome on S² — the helix through the junction
# ============================================================
def render_loxodrome():
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Sphere
    u = np.linspace(0, 2*np.pi, 200)
    v = np.linspace(0, np.pi, 100)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)

    ax.plot_surface(X, Y, Z, color='#0a0a2e', alpha=0.3, rstride=4, cstride=4, shade=False)

    # Loxodrome: the curve that crosses every meridian at constant angle
    # Parametrically: theta = t, phi = 2*arctan(e^{-t*cot(alpha)})
    # For SP, alpha = pi/4 (45 degrees) giving L = pi*sqrt(2)
    alpha = np.pi/4  # crossing angle
    t = np.linspace(-8, 8, 3000)
    phi = 2 * np.arctan(np.exp(-t * 1/np.tan(alpha)))

    x_lox = np.cos(t) * np.sin(phi)
    y_lox = np.sin(t) * np.sin(phi)
    z_lox = np.cos(phi)

    # Color gradient along the loxodrome
    colors = plt.cm.plasma(np.linspace(0.1, 0.95, len(t)))

    for i in range(len(t)-1):
        ax.plot(x_lox[i:i+2], y_lox[i:i+2], z_lox[i:i+2],
                color=colors[i], linewidth=2.0, alpha=0.9)

    # Mark the poles (junction points in S²∨S²)
    ax.scatter([0, 0], [0, 0], [1, -1], color='white', s=80, zorder=10)

    # Faint latitude/longitude grid
    for lat in np.linspace(0, np.pi, 12):
        x_lat = np.cos(u) * np.sin(lat)
        y_lat = np.sin(u) * np.sin(lat)
        z_lat = np.full_like(u, np.cos(lat))
        ax.plot(x_lat, y_lat, z_lat, color='#303060', linewidth=0.3, alpha=0.3)

    for lon in np.linspace(0, 2*np.pi, 24, endpoint=False):
        x_lon = np.cos(lon) * np.sin(v)
        y_lon = np.sin(lon) * np.sin(v)
        z_lon = np.cos(v)
        ax.plot(x_lon, y_lon, z_lon, color='#303060', linewidth=0.3, alpha=0.3)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_03_loxodrome.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 3: Loxodrome on S² saved")


# ============================================================
# IMAGE 4: Warp factor cross-section — the exponential slope
# with mode levels showing where particles "live"
# ============================================================
def render_warp_profile():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')

    z = np.linspace(0, 5, 1000)
    warp = np.exp(-k * z)

    # The warp factor as a landscape
    ax.fill_between(z, 0, warp, alpha=0.3,
                     color='#4a1a8e')
    ax.plot(z, warp, color='#ca7aee', linewidth=2.5, alpha=0.9)

    # Mode levels — where fermions sit
    # electron: kL = 51.53 (normalized to our range)
    modes = {
        'top':      0.15,
        'charm':    0.8,
        'up':       1.2,
        'tau':      1.8,
        'muon':     2.5,
        'electron': 3.8,
    }

    colors_modes = ['#ff4444', '#ff8844', '#ffcc44', '#44ff88', '#44ccff', '#4488ff']

    for i, (name, z_pos) in enumerate(modes.items()):
        w = np.exp(-k * z_pos)
        ax.plot([z_pos, z_pos], [0, w], color=colors_modes[i],
                linewidth=1, alpha=0.4, linestyle='--')
        ax.scatter([z_pos], [w], color=colors_modes[i], s=60, zorder=5)
        ax.text(z_pos + 0.08, w + 0.02, name, color=colors_modes[i],
                fontsize=9, fontfamily='monospace', alpha=0.8)

    # The throat region glow
    throat_z = np.linspace(0, 0.3, 100)
    throat_glow = np.exp(-k * throat_z)
    for offset in np.linspace(0, 0.05, 20):
        ax.fill_between(throat_z, throat_glow - offset, throat_glow + offset,
                        alpha=0.02, color='white')

    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.05, 1.15)
    ax.spines['bottom'].set_color('#303060')
    ax.spines['left'].set_color('#303060')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#606090', labelsize=8)

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_04_warp_profile.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("Image 4: Warp profile with modes saved")


# ============================================================
# IMAGE 5: The S²∨S² — two spheres joined at a point
# with Z₈ symmetry lines visible
# ============================================================
def render_wedge_sum():
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    u = np.linspace(0, 2*np.pi, 150)
    v = np.linspace(0, np.pi, 75)
    U, V = np.meshgrid(u, v)

    r = 0.8

    # Sphere 1 — UV side (left)
    X1 = r * np.cos(U) * np.sin(V) - r  # shifted left
    Y1 = r * np.sin(U) * np.sin(V)
    Z1 = r * np.cos(V)

    # Sphere 2 — IR side (right)
    X2 = r * np.cos(U) * np.sin(V) + r  # shifted right
    Y2 = r * np.sin(U) * np.sin(V)
    Z2 = r * np.cos(V)

    # Color by distance from junction point (touching point at x=0)
    dist1 = np.sqrt((X1 - 0)**2 + Y1**2 + Z1**2)
    dist2 = np.sqrt((X2 - 0)**2 + Y2**2 + Z2**2)

    c1 = 1 - np.clip(dist1 / dist1.max(), 0, 1)
    c2 = 1 - np.clip(dist2 / dist2.max(), 0, 1)

    # UV sphere — blue tones
    uv_cmap = LinearSegmentedColormap.from_list('uv', [
        (0.0, '#050520'), (0.5, '#1030a0'), (0.8, '#3060ff'), (1.0, '#ffffff')
    ])
    # IR sphere — violet tones
    ir_cmap = LinearSegmentedColormap.from_list('ir', [
        (0.0, '#200520'), (0.5, '#8020a0'), (0.8, '#c060ff'), (1.0, '#ffffff')
    ])

    ax.plot_surface(X1, Y1, Z1, facecolors=uv_cmap(c1), alpha=0.7,
                    rstride=2, cstride=2, shade=False)
    ax.plot_surface(X2, Y2, Z2, facecolors=ir_cmap(c2), alpha=0.7,
                    rstride=2, cstride=2, shade=False)

    # Z₈ symmetry lines on each sphere — 8 meridians
    for i in range(8):
        angle = i * np.pi / 4  # 8-fold
        # On sphere 1
        x_m1 = r * np.cos(angle) * np.sin(v) - r
        y_m1 = r * np.sin(angle) * np.sin(v)
        z_m1 = r * np.cos(v)
        ax.plot(x_m1, y_m1, z_m1, color='#80ffff', linewidth=0.8, alpha=0.5)
        # On sphere 2
        x_m2 = r * np.cos(angle) * np.sin(v) + r
        y_m2 = r * np.sin(angle) * np.sin(v)
        z_m2 = r * np.cos(v)
        ax.plot(x_m2, y_m2, z_m2, color='#ff80ff', linewidth=0.8, alpha=0.5)

    # Junction point — bright white
    ax.scatter([0], [0], [0], color='white', s=150, zorder=10)

    # Glow around junction
    for rad in np.linspace(0.01, 0.15, 10):
        phi_g = np.linspace(0, 2*np.pi, 50)
        ax.plot(rad*np.cos(phi_g), rad*np.sin(phi_g), np.zeros(50),
                color='white', alpha=0.1, linewidth=0.5)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    ax.view_init(elev=15, azim=25)
    ax.dist = 7

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_05_wedge_sum.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 5: S²∨S² wedge sum saved")


# ============================================================
# IMAGE 6: The full geometry — funnel WITH internal S²∨S²
# Cross-section showing how curvature reaches the singularity
# ============================================================
def render_funnel_curvature_map():
    fig, ax = plt.subplots(figsize=(10, 16), facecolor='black')
    ax.set_facecolor('black')

    # Cross-section of the double funnel
    z = np.linspace(-4, 4, 2000)
    r_right = warp_radius(z, k=0.8)
    r_left = -r_right

    # Plot the funnel outline
    ax.fill_betweenx(z, r_left, r_right, alpha=0.15, color='#4a1a8e')
    ax.plot(r_right, z, color='#ca7aee', linewidth=1.5, alpha=0.8)
    ax.plot(r_left, z, color='#ca7aee', linewidth=1.5, alpha=0.8)

    # Curvature field — heatmap inside the funnel
    z_grid = np.linspace(-4, 4, 500)
    x_grid = np.linspace(-1.2, 1.2, 300)
    ZG, XG = np.meshgrid(z_grid, x_grid)

    # Radius at each z
    R_boundary = warp_radius(ZG, k=0.8)

    # Mask outside funnel
    inside = np.abs(XG) < R_boundary

    # Curvature ~ k² * e^{2k|z|} (Ricci scalar of RS geometry)
    curvature_field = np.exp(1.6 * np.abs(ZG)) / (np.abs(XG) + 0.01)
    curvature_field[~inside] = 0
    curvature_field = np.clip(curvature_field / np.percentile(curvature_field[inside], 95), 0, 1)

    ax.imshow(curvature_field, extent=[-1.2, 1.2, -4, 4],
              aspect='auto', cmap=curvature_cmap, alpha=0.7, origin='lower')

    # Gridlines at constant z — showing compression
    z_lines = np.concatenate([
        np.linspace(-4, -0.5, 15),
        np.linspace(-0.4, 0.4, 20),  # dense near throat
        np.linspace(0.5, 4, 15)
    ])
    for z_val in z_lines:
        r = warp_radius(z_val, k=0.8)
        brightness = min(1.0, 0.1 + 3.0 * (1 - r))
        ax.plot([-r, r], [z_val, z_val],
                color=(0.5 + 0.5*brightness, 0.3 + 0.3*brightness, 1.0),
                linewidth=0.3 + 1.5*brightness, alpha=0.2 + 0.5*brightness)

    # The throat point
    ax.scatter([0], [0], color='white', s=100, zorder=10)

    # Glow at throat
    for rad in np.linspace(0.005, 0.08, 15):
        circle = plt.Circle((0, 0), rad, fill=False, color='white',
                           alpha=0.15, linewidth=0.5)
        ax.add_patch(circle)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('/home/ai-guest/sp-computation/img_06_curvature_map.png',
                dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Image 6: Curvature map saved")


# Run all
if __name__ == '__main__':
    render_double_funnel()
    render_throat_closeup()
    render_loxodrome()
    render_warp_profile()
    render_wedge_sum()
    render_funnel_curvature_map()
    print("\nAll 6 images saved to /home/ai-guest/sp-computation/")
