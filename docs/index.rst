Freethreading
=============

**Automatic selection between threading and multiprocessing based on Python's GIL status.**

.. image:: https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=LWBRgtlX8j
   :target: https://codecov.io/gh/iskandergaba/freethreading
   :alt: Codecov

.. image:: https://readthedocs.org/projects/freethreading/badge/?version=latest
   :target: https://pyriodicity.readthedocs.io/en/latest
   :alt: Docs

.. image:: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml
   :alt: CI Build

Freethreading provides a unified API for concurrent programming that automatically uses :mod:`threading` when the GIL is disabled (free-threaded Python) and :mod:`multiprocessing` when the GIL is enabled (standard Python). Write your code once and get optimal performance regardless of Python build.

Installation
------------

To install ``freethreading``, simply run:

.. code-block:: shell

   pip install freethreading

To install the latest development version, you can run:

.. code-block:: shell

   pip install git+https://github.com/iskandergaba/freethreading.git


Quick Start
-----------

.. code-block:: python

   import freethreading

   # Check which backend is being used
   print(freethreading.get_backend())  # 'threading' or 'multiprocessing'

   # Use unified API - automatically selects best backend
   from freethreading import Worker, Lock, Queue

   # Create synchronization primitives in main scope
   q = Queue()
   lock = Lock()

   def worker_function():
       """Worker accesses queue and lock from outer scope."""
       with lock:
           q.put("Hello from worker!")

   # Create and start worker
   worker = Worker(target=worker_function)
   worker.start()
   worker.join()
   
   print(q.get())  # 'Hello from worker!'


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
