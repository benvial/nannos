
.. |release_badge| image:: https://img.shields.io/endpoint?url=https://gitlab.com/nannos/nannos/-/jobs/artifacts/master/raw/logobadge.json?job=badge
  :target: https://gitlab.com/nannos/nannos/-/releases
  :alt: Release

.. |GL_CI| image:: https://img.shields.io/gitlab/pipeline/nannos/nannos/master?logo=gitlab&labelColor=grey&style=for-the-badge
  :target: https://gitlab.com/nannos/nannos/commits/master
  :alt: pipeline status

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/nannos?logo=conda-forge&color=CD5C5C&logoColor=white&style=for-the-badge   
  :target: https://anaconda.org/conda-forge/nannos
  :alt: Conda (channel only)

.. |conda_dl| image:: https://img.shields.io/conda/dn/conda-forge/nannos?logo=conda-forge&logoColor=white&style=for-the-badge
  :alt: Conda

.. |conda_platform| image:: https://img.shields.io/conda/pn/conda-forge/nannos?logo=conda-forge&logoColor=white&style=for-the-badge
  :alt: Conda


.. |pip| image:: https://img.shields.io/pypi/v/nannos?color=blue&logo=pypi&logoColor=e9d672&style=for-the-badge
  :target: https://pypi.org/project/nannos/
  :alt: PyPI
  
.. |pip_dl| image:: https://img.shields.io/pypi/dm/nannos?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Downloads
   
.. |pip_status| image:: https://img.shields.io/pypi/status/nannos?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=e9d672&style=for-the-badge
  :alt: Code style: black
 
.. |coverage| image:: https://img.shields.io/gitlab/coverage/nannos/nannos/master?logo=python&logoColor=e9d672&style=for-the-badge
  :target: https://gitlab.com/nannos/nannos/commits/master
  :alt: coverage report 
  
.. |zenodo| image:: https://img.shields.io/badge/DOI-10.5281/zenodo.6490098-dd7d54?logo=google-scholar&logoColor=dd7d54&style=for-the-badge
  :target: https://doi.org/10.5281/zenodo.6490098
 
.. |licence| image:: https://img.shields.io/badge/license-GPLv3-blue?color=dd7d54&logo=open-access&logoColor=dd7d54&style=for-the-badge
  :target: https://gitlab.com/nannos/nannos/-/blob/master/LICENCE.txt
  :alt: license

+----------------------+----------------------+----------------------+
| Release              |            |release_badge|                  |
+----------------------+----------------------+----------------------+
| Deployment           | |pip|                |        |conda|       |
+----------------------+----------------------+----------------------+
| Build Status         |            |GL_CI|                          |
+----------------------+----------------------+----------------------+
| Metrics              |                |coverage|                   |
+----------------------+----------------------+----------------------+
| Activity             |     |pip_dl|         |      |conda_dl|      |
+----------------------+----------------------+----------------------+
| Citation             |           |zenodo|                          |
+----------------------+----------------------+----------------------+
| License              |           |licence|                         |
+----------------------+----------------------+----------------------+
| Formatter            |           |black|                           |
+----------------------+----------------------+----------------------+



.. inclusion-marker-badges

=============================================================
nannos: Fourier Modal Method for multilayer metamaterials
=============================================================


.. inclusion-marker-install-start

Installation
============

From conda
----------

If using `conda <https://www.anaconda.com/>`_, first, add conda-forge to your channels with:

.. code-block:: bash
    
    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the conda-forge channel has been enabled, nannos can be installed with:

.. code-block:: bash
  
  conda install nannos


Alternatively, we provide an `environment.yml <https://gitlab.com/nannos/nannos/-/blob/master/environment.yml>`_ 
file with all the dependencies for the master branch. First create the environment:

.. code-block:: bash

  conda env create -f environment.yml

and then activate it with 

.. code-block:: bash

  conda activate nannos
  

See the `github repository <https://github.com/conda-forge/nannos-feedstock/>`_ 
where development happens for conda-forge.
  

From pypi
---------

The package is available on `pypi <https://pypi.org/project/nannos>`_.
To install, simply use:

.. code-block:: bash

  pip install nannos


From sources
-------------

Sources are available on `gitlab <https://gitlab.com/nannos/nannos>`_. First
clone the repository and install with ``pip``:

.. code-block:: bash

  git clone https://gitlab.com/nannos/nannos.git
  cd nannos
  pip install -e .


.. inclusion-marker-install-end


Documentation
=============

The reference documentation and examples can be found on the
`project website <https://nannos.gitlab.io>`_.


License
=======


.. inclusion-marker-license-start

This software is published under the `GPLv3 license <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.


.. inclusion-marker-license-end
