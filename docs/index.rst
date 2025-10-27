``freethreading`` — GIL-agnostic parallelism
============================================

|Codecov| |Docs| |CI Build|

--------------------------------------------

.. warning::
   This project is a work in progress and is not ready yet for production use.

:mod:`freethreading` is thin layer that provides a unified `Global Interpreter Lock (GIL)
<https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`_ agnostic concurrency API
that ensures true parallel execution of your code regardless of Python and uses `threads
<https://en.wikipedia.org/wiki/Thread_(computing)>`_ instead of `processes
<https://en.wikipedia.org/wiki/Process_(computing)>`_ whenever possible. It does so by automatically
selecting :mod:`threading` for code execution when the GIL is disabled (i.e., free-threaded Python)
and falling back to :mod:`multiprocessing` otherwise (i.e., standard Python).


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

:mod:`freethreading` is an effortless drop-in replacement for pre-existing
:mod:`threading` and :mod:`multiprocessing` code. To achieve this, the module exposes only
non-deprecated common functionality shared between both backends—which happens to be the majority
of available features—while discarding any backend-specific APIs. Here's how to check which
backend is being used:

.. code-block:: pycon

   >>> from freethreading import get_backend
   >>> get_backend()  # 'threading' or 'multiprocessing' depending on your Python build
   'threading'

:mod:`freethreading` remains, for the most part, consistent with the standard library. Here's
an example of how to use :mod:`freethreading` wrapper classes as a drop-in replacement to those 
used by :mod:`threading` and :mod:`multiprocessing`:

.. code-block:: pycon

   >>> # threading
   >>> # from queue import Queue
   >>> # from threading import Lock, Event
   >>>
   >>> # multiprocessing
   >>> # from multiprocessing import Lock, Event, Queue
   >>>
   >>> # freethreading
   >>> from freethreading import Lock, Event, Queue
   >>> lock = Lock()
   >>> event = Event()
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

:mod:`freethreading` wrapper functions merge as much functionality from both backends as
possible to ensure a consistent behavior accross backends and smooth migration. Here's an
example of how to use them as a drop-in replacement to those used by :mod:`threading`
and :mod:`multiprocessing`:

.. code-block:: pycon

   >>> # threading
   >>> # from threading import enumerate, get_ident
   >>>
   >>> # multiprocessing
   >>> # from multiprocessing import active_children
   >>> # from os import getpid
   >>>
   >>> # freethreading
   >>> from freethreading import active_children, enumerate, get_ident
   >>> len(active_children())  # excludes current thread or process
   0
   >>> len(enumerate())  # includes current thread or process
   1
   >>> get_ident()  # current thread or process identifier
   140247834...

Only :func:`~freethreading.current_worker`, :class:`~freethreading.Worker`, and
:class:`~freethreading.WorkerPoolExecutor` differ from the standard library naming,
using "worker" as a unified term for both threads and processes. This example
demonstrates their usage:

.. code-block:: pycon

   >>> # threading
   >>> # from concurrent.futures import ThreadPoolExecutor
   >>> # from threading import current_thread, Thread
   >>>
   >>> # multiprocessing
   >>> # from concurrent.futures import ProcessPoolExecutor
   >>> # from multiprocessing import current_process, Process
   >>>
   >>> # freethreading
   >>> from freethreading import current_worker, Worker, WorkerPoolExecutor
   >>> current_worker().name # 'MainThread' or 'MainProcess'
   'MainThread'
   >>>
   >>> def task():
   ...     print(f"Hello from {current_worker().name}!")
   >>>
   >>> # Using Worker (Thread or Process) to run a task
   >>> w = Worker(target=task, name="MyWorker")
   >>> w.start()
   >>> w.join()
   Hello from MyWorker!
   >>>
   >>> # Using WorkerPoolExecutor (ThreadPoolExecutor or ProcessPoolExecutor) to run a task
   >>> with WorkerPoolExecutor(max_workers=2) as executor:
   ...     future = executor.submit(task)
   >>> # 'Hello from ThreadPoolExecutor-0_0!' or 'Hello from ForkProcess-2!'
   Hello from ThreadPoolExecutor-0_0!

Let's compute factorials in parallel using :class:`~freethreading.WorkerPoolExecutor`.
Under the hood, this automatically uses :class:`~concurrent.futures.ThreadPoolExecutor` on
free-threaded Python and :class:`~concurrent.futures.ProcessPoolExecutor` otherwise,
giving you true parallelism and the efficiency of threads over processes whenever possible.

.. code-block:: pycon

   >>> from freethreading import WorkerPoolExecutor
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


.. |Codecov| image:: https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=D5F3PKSOEK
   :target: https://codecov.io/gh/iskandergaba/freethreading
   :alt: Codecov
.. |Docs| image:: https://readthedocs.org/projects/freethreading/badge/?version=latest
   :target: https://freethreading.readthedocs.io/en/latest
   :alt: Docs
.. |CI Build| image:: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml
   :alt: CI Build
