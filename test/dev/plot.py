#!/usr/bin/env python


import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

import nannos as nn

pv.set_plot_theme("document")

L1 = [1.0, 0]
L2 = [0, 1.0]

Nx = 2 ** 7
Ny = 2 ** 7

eps_pattern = 4.0
eps_hole = 1

x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
lays = [nn.Layer("sup")]


import nannos.optimize as no

density0 = np.random.rand(Nx, Ny)
density0 = np.array(density0)
density0 = 0.5 * (density0 + np.fliplr(density0))
density0 = 0.5 * (density0 + np.flipud(density0))
density0 = 0.5 * (density0 + density0.T)
density0 = no.apply_filter(density0, Nx / 20)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())

density0[density0 < 0.3] = 0
density0[np.logical_and(density0 >= 0.3, density0 < 0.6)] = 0.5
density0[density0 >= 0.6] = 1


epsgrid = no.simp(density0, 1, 11, p=1)
pattern = nn.Pattern(epsgrid, name="design")
meta = nn.Layer("meta", thickness=0.2)
meta.add_pattern(pattern)
lays.append(meta)


il = 0
for radius, thickness in zip([0.3, 0.2, 0.1], [0.3, 0.7, 0.1]):
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2
    ids = np.ones((Nx, Ny), dtype=float)
    epsgrid = ids * (np.random.rand(1) * 10 + 2)
    epsgrid[hole] = eps_hole
    st = nn.Layer(f"pat{il}", thickness=thickness)
    st.add_pattern(nn.Pattern(epsgrid))
    lays.append(st)
    il += 1


lays.append(nn.Layer("sub", epsilon=8))


pw = nn.PlaneWave(1.2)
sim = nn.Simulation(
    nn.Lattice(((1, 0), (0, 1))), lays, pw, nh=200, formulation="tangent"
)


colors = ["#ed7559", "#4589b5", "#cad45f", "#7a6773", "#ed59da"]

p = pv.Plotter()

dz = 0.2

z = 0
for layer in np.flipud(sim.layers):
    thickness = layer.thickness
    if thickness == 0:
        if layer.epsilon.real != 1:
            thickness = 3
    if layer.is_uniform:
        if thickness != 0:
            grid = pv.UniformGrid()
            grid.dimensions = (2, 2, 2)
            grid.origin = (0, 0, z)  # The bottom left corner of the data set
            grid.spacing = (1, 1, thickness)  # These are the cell sizes along each axis
            grid.cell_data["values"] = np.array(
                [layer.epsilon.real]
            )  # Flatten the array!
            # grid.plot()
            mesh = grid.extract_surface()
            p.add_mesh(
                mesh,
                metallic=0.3,
                roughness=0.1,
                pbr=True,
                diffuse=1,
                color=np.random.rand(3),
            )
    else:
        epsgrid = layer.patterns[0].epsilon.real
        # values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
        values = np.reshape(epsgrid, (Nx, Ny, 1))
        # Create the spatial reference
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0, 0, z)  # The bottom left corner of the data set
        grid.spacing = (
            1 / Nx,
            1 / Ny,
            thickness,
        )  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        vals = np.unique(epsgrid)
        for v in vals:
            if v != 1:
                threshed = grid.threshold([v - 1e-7 * v, v + 1e-7 * v])
                p.add_mesh(
                    threshed,
                    metallic=0.3,
                    roughness=0.1,
                    pbr=True,
                    diffuse=1,
                    color=np.random.rand(3),
                    opacity=1,
                )

    z += thickness + dz
p.show()


sx
# p.add_mesh(threshed, cmap=my_colormap,
#            pbr=True,lighting='three lights')
# light = pv.Light()
# light.set_direction_angle(70, -50)
# p.add_light(light)


# # Now plot the grid!
# grid.plot(show_edges=False, color='linen',
#            pbr=True, metallic=0.8, roughness=0.1,
#            diffuse=1)
# Define the colors we want to use
blue = np.array([12 / 256, 33 / 256, 200 / 256, 1])
grey = np.array([189 / 256, 189 / 256, 189 / 256, 1])
mapping = np.linspace(grid["values"].min(), grid["values"].max(), 2)
newcolors = np.empty((2, 4))
newcolors[mapping == eps_hole] = blue
newcolors[mapping == eps_pattern] = grey

# Make the colormap from the listed colors
my_colormap = ListedColormap(newcolors)

#
#
# # Define a nice camera perspective
# cpos = [(-313.40, 66.09, 1000.61),
#         (0.0, 0.0, 0.0),
#         (0.018, 0.99, -0.06)]


# Apply a threshold over a data range

p = pv.Plotter()

# def plot_lay()

threshed = grid.threshold([0, 1])
# p.add_mesh(threshed, cmap=my_colormap,
#            pbr=True,lighting='three lights')
threshed = grid.threshold([eps_pattern, 111])
# light = pv.Light()
# light.set_direction_angle(70, -50)
# p.add_light(light)
p.add_mesh(threshed, color="#77a38d", metallic=0.3, roughness=0.1, pbr=True, diffuse=1)
p.show()

# grid.plot(show_edges=False, cmap=my_colormap,
#            pbr=True)

# from pyvista import examples
# cubemap = examples.download_sky_box_cube_map()

# p = pv.Plotter()
# p.add_actor(cubemap.to_skybox())
# p.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
