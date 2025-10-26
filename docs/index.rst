``freethreading`` — GIL-agnostic parallelism
============================================

|Codecov| |Docs| |CI Build|

--------------------------------------------

.. warning::
   This project is a work in progress and is not ready yet for production use.

``freethreading`` provides a unified Global Interpreter Lock (GIL) agnostic concurrency API that
automatically selects :mod:`threading` for execution when the GIL is disabled (i.e., free-threaded
Python) and falls back to :mod:`multiprocessing` otherwise. ``freethreading`` allows you to write
your code once and get optimal performance regardless of Python build, always bypassing the GIL
and using :mod:`threading` whenever possible.

Installation
------------

.. To install ``freethreading``, simply run:

.. .. code-block:: shell

..    pip install freethreading

To install the latest development version, you can run:

.. code-block:: shell

   pip install git+https://github.com/iskandergaba/freethreading.git


Quick Start
-----------

.. code-block:: pycon

   >>> from freethreading import get_backend, WorkerPoolExecutor
   >>>
   >>> # Check which backend is being used
   >>> print(get_backend())
   threading # or 'multiprocessing' depending on Python build
   >>>
   >>> def factorial(n):
   ...     """Compute factorial of n."""
   ...     result = 1
   ...     for i in range(2, n + 1):
   ...         result *= i
   ...     return result
   ...
   >>> # Compute factorials in parallel using WorkerPoolExecutor
   >>> numbers = [5, 10, 15, 20]
   >>> with WorkerPoolExecutor(max_workers=4) as executor:
   ...     results = list(executor.map(factorial, numbers))
   ...
   >>> # Display results
   >>> for num, fact in zip(numbers, results):
   ...     print(f"{num}! = {fact}")
   ...
   5! = 120
   10! = 3628800
   15! = 1307674368000
   20! = 2432902008176640000


Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   caveats
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Codecov| image:: https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=D5F3PKSOEK
   :target: https://codecov.io/gh/iskandergaba/freethreading
   :alt: Codecov
.. |Docs| image:: https://readthedocs.org/projects/freethreading/badge/?version=latest
   :target: https://freethreading.readthedocs.io/en/latest
   :alt: Docs
.. |CI Build| image:: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml
   :alt: CI Build
