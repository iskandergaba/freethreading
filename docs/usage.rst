Usage Guide
===========

Basic Concepts
--------------

Freethreading provides a unified API for concurrent programming that adapts to your Python runtime:

- **With GIL enabled** (standard Python): Uses :mod:`multiprocessing` for true parallelism
- **With GIL disabled** (free-threaded Python 3.13+): Uses :mod:`threading` for lower overhead

This automatic selection happens at import time, so your code remains identical across different Python builds.

Checking Backend
----------------

You can determine which backend is active:

.. code-block:: python

   import freethreading

   # Get backend name directly
   backend = freethreading.get_backend()  # 'threading' or 'multiprocessing'

   if backend == 'multiprocessing':
       print("GIL is enabled - using multiprocessing")
   else:
       print("GIL is disabled - using threading")


Workers (Threads/Processes)
----------------------------

The :class:`~freethreading.Worker` class provides a unified interface for creating threads or processes:

.. code-block:: python

   from freethreading import Worker

   def task(name, count):
       for i in range(count):
           print(f"{name}: {i}")

   # Create and start worker
   w = Worker(target=task, args=("Worker-1", 5))
   w.start()
   w.join()

   # With daemon workers
   w = Worker(target=task, args=("Background", 10), daemon=True)
   w.start()


Synchronization Primitives
---------------------------

All synchronization primitives work identically across backends:

Locks
^^^^^

.. code-block:: python

   from freethreading import Lock

   lock = Lock()

   def critical_section():
       with lock:
           # Thread/process-safe code
           print("In critical section")

RLocks
^^^^^^

.. code-block:: python

   from freethreading import RLock

   rlock = RLock()

   def recursive_function(n):
       with rlock:
           if n > 0:
               recursive_function(n - 1)

Semaphores
^^^^^^^^^^

.. code-block:: python

   from freethreading import Semaphore, BoundedSemaphore

   # Standard semaphore
   sem = Semaphore(3)  # Allow 3 concurrent accesses

   def limited_resource():
       with sem:
           # Only 3 workers can be here at once
           print("Using resource")

   # Bounded semaphore (prevents over-releasing)
   bounded = BoundedSemaphore(2)

Conditions
^^^^^^^^^^

.. code-block:: python

   from freethreading import Condition, Worker

   condition = Condition()
   data = []

   def producer():
       with condition:
           data.append("item")
           condition.notify()

   def consumer():
       with condition:
           condition.wait()
           item = data.pop()
           print(f"Consumed: {item}")

Events and Barriers
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from freethreading import Event, Barrier, Worker

   # Events
   event = Event()

   def waiter():
       print("Waiting for event...")
       event.wait()
       print("Event set!")

   def setter():
       import time
       time.sleep(1)
       event.set()

   # Barriers
   barrier = Barrier(3)  # Wait for 3 workers

   def synchronized_task(i):
       print(f"Worker {i} reached barrier")
       barrier.wait()
       print(f"Worker {i} past barrier")


Queues
------

Use queues for safe data passing between workers:

Queue
^^^^^

.. code-block:: python

   from freethreading import Queue, Worker

   q = Queue(maxsize=10)

   def producer():
       for i in range(5):
           q.put(i)
       q.put(None)  # Sentinel

   def consumer():
       while True:
           item = q.get()
           if item is None:
               q.task_done()
               break
           print(f"Processing: {item}")
           q.task_done()

   p = Worker(target=producer)
   c = Worker(target=consumer)
   p.start()
   c.start()
   q.join()  # Wait for all tasks to complete
   p.join()
   c.join()

SimpleQueue
^^^^^^^^^^^

.. code-block:: python

   from freethreading import SimpleQueue

   sq = SimpleQueue()
   sq.put("Hello")
   sq.put("World")

   print(sq.get())  # 'Hello'
   print(sq.empty())  # False


Executor Pattern
----------------

For higher-level parallelism, use :class:`~freethreading.WorkerPoolExecutor`:

.. code-block:: python

   from freethreading import WorkerPoolExecutor

   def compute(x):
       return x * x

   # Automatically uses ThreadPoolExecutor or ProcessPoolExecutor
   with WorkerPoolExecutor(max_workers=4) as executor:
       results = executor.map(compute, range(10))
       print(list(results))


Utility Functions
-----------------

Inspection Functions
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from os import cpu_count
   from freethreading import (
       active_count,
       current_worker,
       enumerate,
       get_ident,
   )

   print(f"CPUs: {cpu_count()}")
   print(f"Active workers: {active_count()}")
   print(f"Current worker: {current_worker()}")
   print(f"Worker ID: {get_ident()}")

   # List all active workers
   for worker in enumerate():
       print(f"Worker: {worker.name}")


Example: Parallel Computing
----------------------------

Here's a complete example computing prime numbers:

.. code-block:: python

   from os import cpu_count
   from freethreading import Worker, Queue

   def is_prime(n):
       if n < 2:
           return False
       for i in range(2, int(n ** 0.5) + 1):
           if n % i == 0:
               return False
       return True

   # Create queues in main scope
   input_q = Queue()
   output_q = Queue()

   def worker():
       """Worker function that accesses queues from outer scope."""
       while True:
           n = input_q.get()
           if n is None:
               input_q.task_done()
               break
           if is_prime(n):
               output_q.put(n)
           input_q.task_done()

   # Start workers
   workers = []
   for _ in range(cpu_count()):
       w = Worker(target=worker)
       w.start()
       workers.append(w)

   # Feed work
   for i in range(1000, 2000):
       input_q.put(i)

   # Signal workers to stop
   for _ in workers:
       input_q.put(None)

   # Wait for completion
   input_q.join()
   for w in workers:
       w.join()

   # Collect results
   primes = []
   while not output_q.empty():
       primes.append(output_q.get())

   print(f"Found {len(primes)} primes")
