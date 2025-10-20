Caveats & Limitations
=====================

Understanding when and how to use ``freethreading`` is crucial for building reliable concurrent applications.

Who Should Use This Library
----------------------------

✅ **Good fit for:**

- Projects targeting both standard and free-threaded Python builds
- CPU-bound parallel workloads that need to work across Python versions
- Applications where you want threading's low overhead when GIL is disabled, but need multiprocessing fallback
- Libraries that want to provide automatic optimization based on runtime

❌ **Not recommended for:**

- Projects that only target free-threaded Python (use :mod:`threading` directly)
- Projects that only target standard Python (use :mod:`multiprocessing` directly)
- Applications with complex shared state (see limitations below)
- Real-time systems requiring predictable behavior
- Projects needing features beyond the common subset of threading/multiprocessing


Key Limitations
---------------

Picklability Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the multiprocessing backend requires serialization, **all data passed to workers must be picklable**:

.. code-block:: python

   from freethreading import Worker

   # ❌ BAD: Lambdas and local functions aren't picklable
   worker = Worker(target=lambda x: x * 2, args=(5,))

   # ✅ GOOD: Module-level functions are picklable
   def double(x):
       return x * 2

   worker = Worker(target=double, args=(5,))

The library will warn you at runtime when it detects non-picklable arguments:

.. code-block:: python

   RuntimeWarning: Arguments may not be picklable: Can't pickle <lambda>
   This will cause issues with multiprocessing backend.
   Use module-level functions instead of lambdas.


Limited Feature Set
^^^^^^^^^^^^^^^^^^^

Only features available in **both** :mod:`threading` and :mod:`multiprocessing` are exposed:

**Included:**

- Basic synchronization: Lock, RLock, Semaphore, BoundedSemaphore
- Coordination: Event, Barrier, Condition
- Communication: Queue, SimpleQueue
- Workers: Thread/Process with common interface
- Utilities: active_count, enumerate, current_worker, get_ident, cpu_count

**Not included:**

- Thread-local/process-local storage
- Thread/process pooling (use :class:`~freethreading.PoolExecutor` instead)
- Manager objects (multiprocessing-specific)
- Shared memory types (multiprocessing-specific)
- Timers (threading-specific)
- Thread-specific setters (setprofile, settrace)


Performance Considerations
--------------------------

Startup Overhead
^^^^^^^^^^^^^^^^

Process creation (multiprocessing) has **significant overhead** compared to threads:

.. code-block:: python

   # With GIL disabled (threading): microseconds
   # With GIL enabled (multiprocessing): milliseconds

   # Bad: Creating many short-lived workers
   for i in range(1000):
       w = Worker(target=small_task)
       w.start()
       w.join()

   # Better: Use PoolExecutor for many tasks
   from freethreading import PoolExecutor

   with PoolExecutor(max_workers=4) as executor:
       executor.map(small_task, range(1000))


SimpleQueue Limitations
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~freethreading.SimpleQueue` has behavioral differences:

- In **threading** mode: Supports ``block`` and ``timeout`` parameters
- In **multiprocessing** mode: Always blocks (parameters ignored with warning)

.. code-block:: python

   from freethreading import SimpleQueue

   sq = SimpleQueue()

   # This works in threading mode but warns in multiprocessing mode
   sq.put("item", block=False, timeout=1)


GIL Status Detection
--------------------

The library checks GIL status at **import time**:

.. code-block:: python

   import freethreading

   # Backend is determined once, at import
   # Changing sys._is_gil_enabled() after import has no effect

This means:

- Backend selection is consistent throughout application lifetime
- No runtime overhead checking GIL status repeatedly
- Can't switch backends dynamically


Debugging Tips
--------------

Identifying Backend Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

When debugging, check which backend is active:

.. code-block:: python

   import freethreading

   print(f"Backend: {freethreading.get_backend()}")
   print(f"GIL enabled: {freethreading.is_gil_enabled()}")

Force a specific backend for testing:

.. code-block:: python

   # Test with multiprocessing behavior
   import sys
   sys._is_gil_enabled = lambda: True  # Must be done before import

   import freethreading
   # Now uses multiprocessing backend


Common Pitfalls
^^^^^^^^^^^^^^^

1. **Shared State**

   .. code-block:: python

      # ❌ BAD: Global state won't work with multiprocessing
      counter = 0

      def increment():
          global counter
          counter += 1

      # ✅ GOOD: Use Queue or Manager
      from freethreading import Queue

      q = Queue()

      def increment(q):
          current = q.get()
          q.put(current + 1)

2. **Resource Cleanup**

   .. code-block:: python

      # ❌ BAD: Resources might not be cleaned up
      worker = Worker(target=task)
      worker.start()
      # ... program exits without join()

      # ✅ GOOD: Always join workers
      worker = Worker(target=task)
      worker.start()
      try:
          worker.join()
      finally:
          # cleanup

3. **Daemon Workers**

   .. code-block:: python

      # Daemon workers are killed at program exit
      # Don't use for critical work

      worker = Worker(target=important_task, daemon=True)
      worker.start()
      # May not complete if program exits


Migration Guide
---------------

From threading
^^^^^^^^^^^^^^

.. code-block:: python

   # Before
   import threading

   lock = threading.Lock()
   thread = threading.Thread(target=func)

   # After
   import freethreading

   lock = freethreading.Lock()
   thread = freethreading.Worker(target=func)

From multiprocessing
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Before
   import multiprocessing

   lock = multiprocessing.Lock()
   process = multiprocessing.Process(target=func)

   # After
   import freethreading

   lock = freethreading.Lock()
   process = freethreading.Worker(target=func)


Testing Your Code
-----------------

Test both backends:

.. code-block:: python

   # test_myapp.py
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

   def test_my_feature(backend):
       # Test works with both backends
       worker = backend.Worker(target=my_function)
       worker.start()
       worker.join()


When NOT to Use Freethreading
------------------------------

Consider alternatives if:

1. **You need full feature sets:** Use :mod:`threading` or :mod:`multiprocessing` directly
2. **Performance is critical:** The abstraction has small overhead
3. **You need shared memory:** Use :mod:`multiprocessing` with Manager/shared memory
4. **Complex communication patterns:** Consider ``concurrent.futures`` or ``asyncio``
5. **You need thread-local storage:** Not available in unified API

**Remember:** Freethreading is about **portability** across Python builds, not replacing specialized concurrent programming patterns.
