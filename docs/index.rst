``freethreading`` — Thread-first true parallelism
=================================================

|Codecov| |Docs| |CI Build|

--------------------------------------------

.. warning::
   This project is a work in progress and is not ready yet for production use.

:mod:`freethreading` is a lightweight wrapper that provides a unified API for true parallel execution in Python. It
automatically uses :mod:`threading` on free-threaded Python builds (where the `Global Interpreter Lock (GIL)
<https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`_ is disabled) and falls back to
:mod:`multiprocessing` on standard ones. This enables true parallelism across Python versions while preferring the
efficiency of `threads <https://en.wikipedia.org/wiki/Thread_(computing)>`_ over `processes
<https://en.wikipedia.org/wiki/Process_(computing)>`_ whenever possible.


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

:mod:`freethreading` is a drop-in replacement for *most* pre-existing :mod:`threading` and :mod:`multiprocessing` code.
To achieve this, the module exposes only non-deprecated common functionality shared between both backends while
discarding any backend-specific APIs. The following examples show how to get started.

:mod:`freethreading` remains consistent with the standard library, so wrapper classes work as drop-in replacements for
those used by :mod:`threading` and :mod:`multiprocessing`. Here's how they work:

.. code-block:: pycon

   >>> # threading
   >>> from queue import Queue
   >>> from threading import Event, Lock
   >>>
   >>> # multiprocessing
   >>> from multiprocessing import Event, Lock, Queue
   >>>
   >>> # freethreading (replaces both)
   >>> from freethreading import Event, Lock, Queue
   >>> event = Event()
   >>> lock = Lock()
   >>> queue = Queue()
   >>> lock.acquire()
   True
   >>> event.set()
   >>> queue.put("data")
   >>> event.is_set()
   True
   >>> queue.get()
   'data'
   >>> lock.release()

:mod:`freethreading` functions merge as much functionality from both backends as possible to ensure consistent behavior
across backends and simplify adoption. Here's what that looks like:

.. code-block:: pycon

   >>> # threading
   >>> from threading import enumerate, get_ident
   >>>
   >>> # multiprocessing
   >>> from multiprocessing import active_children
   >>> from os import getpid
   >>>
   >>> # freethreading (replaces both)
   >>> from freethreading import active_children, enumerate, get_ident
   >>> len(active_children())  # excludes current thread or process
   0
   >>> len(enumerate())  # includes current thread or process
   1
   >>> get_ident()  # current thread or process identifier
   140247834...

Only :func:`~freethreading.current_worker`, :class:`~freethreading.Worker`, and
:class:`~freethreading.WorkerPoolExecutor` differ from the standard library naming, using "worker" as a unified term
for both threads and processes. Here is an example:

.. code-block:: pycon

   >>> # threading
   >>> from concurrent.futures import ThreadPoolExecutor
   >>> from threading import Thread, current_thread
   >>>
   >>> # multiprocessing
   >>> from concurrent.futures import ProcessPoolExecutor
   >>> from multiprocessing import Process, current_process
   >>>
   >>> # freethreading (replaces both)
   >>> from freethreading import Worker, WorkerPoolExecutor, current_worker
   >>> current_worker().name # 'MainThread' or 'MainProcess'
   'MainThread'
   >>>
   >>> def task():
   ...     print(f"Hello from {current_worker().name}!")
   ...
   >>> # Using Worker (Thread or Process) to run a task
   >>> w = Worker(target=task, name="MyWorker")
   >>> w.start()
   >>> w.join()
   Hello from MyWorker!
   >>>
   >>> # Using WorkerPoolExecutor (ThreadPoolExecutor or ProcessPoolExecutor) to run a task
   >>> with WorkerPoolExecutor(max_workers=2) as executor:
   ...     # 'Hello from ThreadPoolExecutor-0_0!' or 'Hello from ForkProcess-2!'
   ...     future = executor.submit(task)
   >>>
   Hello from ThreadPoolExecutor-0_0!


Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   considerations
   API Reference <generated/freethreading/index>


.. |Codecov| image:: https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=D5F3PKSOEK
   :target: https://codecov.io/gh/iskandergaba/freethreading
   :alt: Codecov
.. |Docs| image:: https://readthedocs.org/projects/freethreading/badge/?version=latest
   :target: https://freethreading.readthedocs.io/en/latest
   :alt: Docs
.. |CI Build| image:: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml
   :alt: CI Build
