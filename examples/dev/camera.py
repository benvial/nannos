#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()

bk = nn.backend
formulation = "tangent"
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**10)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=2)
epsilon = lattice.ones() * 1
hole = lattice.ellipse(center=(0.5, 0.5), radii=(0.2, 0.4))
epsilon[hole] = 2
ms = lattice.Layer("Metasurface", thickness=0.1, epsilon=epsilon)
pw = nn.PlaneWave(wavelength=1 / 1.4, angles=(0, 0, 0 * nn.pi / 2))
nh = 200
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)

layer_colors = [None, ["white", "orange"], "teal"]
layer_metallic = [None, [0, 0.8], 0]
layer_roughness = [None, [0, 0.2], 1]

import pyvista as pv

# from pyvista import examples
# cubemap = examples.download_sky_box_cube_map()


pl = pv.Plotter(lighting="none")

# pl.add_actor(cubemap.to_skybox())
# pl.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
pl = sim.plot_structure(
    nper=(1, 1),
    plotter=pl,
    pbr=True,
    layer_colors=layer_colors,
    layer_metallic=layer_metallic,
    layer_roughness=layer_roughness,
    diffuse=1,
)  # , roughness=0.1, diffuse=1)
#

light = pv.Light(position=(-1, -1, -1), light_type="scene light")
# light.set_direction_angle(0, 0)
# these don't do anything for a headlight:
# light.position = (1, 2, 3)
# light.focal_point = (4, 5, 6)
pl.add_light(light)

light = pv.Light(position=(1, -1, -1), light_type="scene light")
pl.add_light(light)
light = pv.Light(position=(1, 1, -1), light_type="scene light")
pl.add_light(light)

# pl.enable_shadows()

pl.show()
