Important Considerations
========================

Before using :mod:`freethreading`, it is important to understand when it is the right choice and what to watch out for.

When to Use ``freethreading``
-----------------------------

:mod:`freethreading` solves a specific problem: achieving true parallel execution across both standard and
free-threaded Python builds with one codebase. However, this portability comes at the cost of a reduced feature set and
constraints on how workers share data. With that in mind, :mod:`freethreading` works well for:

- Projects targeting both standard and free-threaded Python builds
- Projects whose needs fall within the common feature set of :mod:`threading` and :mod:`multiprocessing`
- Computationally intensive code that benefits from true parallelism on standard Python builds, with the
  added advantage of :mod:`threading`'s lower overhead on free-threaded builds

For anything else, :mod:`threading` or :mod:`multiprocessing` are likely better choices. If a project already relies on
backend-specific features like shared memory or thread-local storage, then the choice is either to use the
corresponding backend or to adapt the code to the common feature set of both.


Platform Limitations
--------------------

.. todo::

   Document macOS limitation: ``Queue.qsize()`` raises ``NotImplementedError`` on macOS with the multiprocessing
   backend because ``sem_getvalue()`` is not implemented on that platform.


Constraints and Pitfalls
------------------------

In addition to general concurrency pitfalls like proper resource cleanup and daemon behavior, :mod:`freethreading` has
a few of its own that stem from supporting consistent behavior across both :mod:`threading` and :mod:`multiprocessing`
backends.

REPL Limitations on Python 3.14
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On Python 3.14 (standard builds), examples using user-defined functions with :class:`~freethreading.Worker`,
:class:`~freethreading.WorkerPool`, or :class:`~freethreading.WorkerPoolExecutor` must be saved to a ``.py`` file to
run.

Picklability Requirement
^^^^^^^^^^^^^^^^^^^^^^^^

Since the :mod:`multiprocessing` backend requires serialization, data passed to workers must be `picklable
<https://docs.python.org/3/library/pickle.html>`_. The library validates this at :class:`~freethreading.Worker`
creation time, ensuring code works consistently regardless of which backend is active. Here's a quick example:

.. code-block:: pycon

   >>> from freethreading import Worker
   >>>
   >>> # This raises ValueError - lambdas aren't picklable
   >>> worker = Worker(target=lambda x: x * 2, args=(5,))
   Traceback (most recent call last):
     ...
   ValueError: Worker arguments must be picklable for compatibility with multiprocessing backend...

Module-level functions are picklable and work with both backends. For instance:

.. code-block:: pycon

   >>> from freethreading import Worker
   >>>
   >>> def double(x):
   ...     return x * 2
   ...
   >>> worker = Worker(target=double, args=(5,))
   >>> worker.start()
   >>> worker.join()


Shared State
^^^^^^^^^^^^

Sharing state across workers through variables may produce unexpected results. Variables defined outside a worker's
function are handled differently depending on the backend — with :mod:`multiprocessing`, each process has its own
memory space, so changes made in one worker are not reflected in other workers. Here is an example:

.. code-block:: pycon

   >>> from freethreading import Worker, get_backend
   >>>
   >>> get_backend()
   'multiprocessing'
   >>>
   >>> counter = 0
   >>>
   >>> def increment():
   ...     global counter
   ...     counter += 1
   ...
   >>> workers = [Worker(target=increment) for _ in range(5)]
   >>> for w in workers:
   ...     w.start()
   >>> for w in workers:
   ...     w.join()
   >>> counter  # Still 0 with multiprocessing!
   0

:class:`~freethreading.Queue` or :class:`~freethreading.SimpleQueue` should be used instead for passing data between
workers. Here is an example:

.. code-block:: pycon

   >>> from freethreading import Queue, Worker, get_backend
   >>>
   >>> get_backend()
   'multiprocessing'
   >>>
   >>> results = Queue()
   >>>
   >>> def increment(results):
   ...     results.put(1)
   ...
   >>> workers = [Worker(target=increment, args=(results,)) for _ in range(5)]
   >>> for w in workers:
   ...     w.start()
   >>> for w in workers:
   ...     w.join()
   >>> total = sum(results.get() for _ in range(5))
   >>> total
   5


Development Tips
----------------

:mod:`freethreading` aims for consistent behavior across backends, but understanding the underlying runtime can help
when investigating issues and validating code. Below are a few tips that can help during development.

Checking the Backend
^^^^^^^^^^^^^^^^^^^^

Knowing which parallelism backend is being used can be helpful for debugging. Here's how to check it:

.. code-block:: pycon

   >>> from freethreading import get_backend
   >>>
   >>> get_backend() # 'threading' or 'multiprocessing' depending on your Python build
   'threading'


Testing Both Backends
^^^^^^^^^^^^^^^^^^^^^

Testing code against both backends ensures it works regardless of which one :mod:`freethreading` selects. It is a great
way to catch some of the pitfalls mentioned above. Here is an example of how to do this using :mod:`pytest`:

.. code-block:: python

   import pytest
   import sys

   @pytest.fixture(params=['threading', 'multiprocessing'])
   def backend(request, monkeypatch):
       if request.param == 'threading':
           monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False)
       else:
           monkeypatch.setattr(sys, '_is_gil_enabled', lambda: True)

       # Clear module cache to re-import with new GIL status
       if 'freethreading' in sys.modules:
           del sys.modules['freethreading']

       import freethreading
       return freethreading

   def test_worker(backend):
       def task():
           pass

       worker = backend.Worker(target=task)
       worker.start()
       worker.join()

And here is an equivalent example using :mod:`unittest`:

.. code-block:: python

   import sys
   import unittest

   class BackendTestCase(unittest.TestCase):
       backend = None
       original_gil_enabled = None

       @classmethod
       def setUpClass(cls):
           cls.original_gil_enabled = getattr(sys, '_is_gil_enabled', None)

           if cls.backend == 'threading':
               sys._is_gil_enabled = lambda: False
           else:
               sys._is_gil_enabled = lambda: True

           if 'freethreading' in sys.modules:
               del sys.modules['freethreading']

           import freethreading
           cls.freethreading = freethreading

       @classmethod
       def tearDownClass(cls):
           if cls.original_gil_enabled is None:
               if hasattr(sys, '_is_gil_enabled'):
                   delattr(sys, '_is_gil_enabled')
           else:
               sys._is_gil_enabled = cls.original_gil_enabled

       def test_worker(self):
           def task():
               pass

           worker = self.freethreading.Worker(target=task)
           worker.start()
           worker.join()

   class TestThreadingBackend(BackendTestCase):
       backend = 'threading'

   class TestMultiprocessingBackend(BackendTestCase):
       backend = 'multiprocessing'
