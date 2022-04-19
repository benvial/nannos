Tutorials
-----------



Basic Example
================

This basic example demonstrates key features of nannos.

.. thebe-button:: Click me to execute code!

First import the package.

.. jupyter-execute::

  import nannos as nn

Create the lattice.

.. jupyter-execute::

  lattice = nn.Lattice(basis_vectors = [[1.0, 0], [0, 1.0]], discretization=2**9)


Define the layers. The thickness for the semi infinite input and output media is irrelevant.

.. jupyter-execute::

  sup = lattice.Layer("Superstrate", epsilon=1)
  ms = lattice.Layer("Metasurface", thickness=0.5)
  sub = lattice.Layer("Substrate", epsilon=2)


Create the permittivity pattern for the structured layer.

.. jupyter-execute::

  ms.epsilon = lattice.ones() * 12.0
  circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
  ms.epsilon[circ] = 1


Define the incident plane wave.

.. jupyter-execute::

  pw = nn.PlaneWave(wavelength=0.9, angles=(0, 0, 0))


Define the layer stack as a list from input medium to output medium.

.. jupyter-execute::

  stack = [sup, ms, sub]


Create the simulation.

.. jupyter-execute::

  sim = nn.Simulation(stack, pw, nh=200)


Plot the unit cell.


.. jupyter-execute::

  p = sim.plot_structure()

.. The pythreejs backend seems to have issues for building the doc so we use panel instead here
Render it.

.. jupyter-execute::

  p.show_axes()
  p.show(jupyter_backend='panel')


Compute the reflection and transmission:

.. jupyter-execute::

  R,T = sim.diffraction_efficiencies()
  print("reflection: ", R)
  print("transmission: ", T)
  print("sum :", R + T)

.. raw:: html

 <p>
 </p>



Other Tutorials
================
