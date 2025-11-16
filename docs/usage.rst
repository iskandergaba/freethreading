Usage Guide
===========

Overview
--------

:mod:`freethreading` shields you from differences between GIL-enabled Python builds and
free-threaded builds. At import time it decides whether to route calls through
:mod:`multiprocessing` (GIL-enabled Python) or :mod:`threading` (GIL-disabled Python).
The public surface mirrors the common subset shared by both backends, so you can write
portable code once and run it efficiently everywhere.


Checking the Active Backend
---------------------------

Confirm the backend that :mod:`freethreading` selected so you can reason about behavior such as process semantics:

.. code-block:: pycon

   >>> from freethreading import get_backend
   >>> get_backend()  # 'threading' or 'multiprocessing' depending on your Python build
   'threading'


Parallel Execution
------------------

:mod:`freethreading` offers low-level :class:`~freethreading.Worker` objects for direct task control
and high-level :class:`~freethreading.WorkerPoolExecutor` objects for managed execution.

``Worker``
^^^^^^^^^^

The :class:`~freethreading.Worker` class represents an activity that is run in a separate thread or process.
It is wrapper that carries over the shared controls from :class:`~threading.Thread` and
:class:`~multiprocessing.Process`, so that you can name workers, set :attr:`~freethreading.Worker.daemon`, and call
familiar methods like :func:`~freethreading.Worker.start()`, :func:`~freethreading.Worker.join()`, and
:func:`~freethreading.Worker.is_alive()`, without worrying about the underlying implementation. Here's a quick example:

.. code-block:: pycon

   >>> from freethreading import Worker, current_worker
   >>>
   >>> def greet():
   ...     print(f"Hello from '{current_worker().name}'")
   ...
   >>> worker = Worker(target=greet, name="Worker")
   >>> worker.start()
   >>> worker.join()
   Hello from 'Worker'
   >>> daemon = Worker(target=greet, name="Daemon", daemon=True)
   >>> daemon.start()
   >>> daemon.join()
   Hello from 'Daemon'

Set :attr:`~freethreading.Worker.daemon` to ``True`` when you need a fire-and-forget background worker,
then invoke methods like :func:`~freethreading.Worker.is_alive()` and :func:`~freethreading.Worker.join()`,
and check attributes such as :attr:`~freethreading.Worker.name`, exactly as you would with Python's standard
:mod:`threading` or :mod:`multiprocessing` modules.

``WorkerPoolExecutor``
^^^^^^^^^^^^^^^^^^^^^^

:class:`~freethreading.WorkerPoolExecutor` provides a unified drop-in replacement for
:class:`~concurrent.futures.ThreadPoolExecutor` and :class:`~concurrent.futures.ProcessPoolExecutor`. For instance:

.. code-block:: pycon

   >>> from freethreading import WorkerPoolExecutor
   >>>
   >>> def square(x):
   ...     return x * x
   ...
   >>> with WorkerPoolExecutor(max_workers=4) as executor:
   ...     futures = executor.map(square, range(5))
   ...     print(list(futures))
   ...
   [0, 1, 4, 9, 16]


Data Exchange
-------------

Workers can exchange data through :class:`~freethreading.Queue` for structured coordination or
:class:`~freethreading.SimpleQueue` for lightweight messaging.

``Queue``
^^^^^^^^^

:class:`freethreading.Queue` wraps :class:`queue.Queue` and :class:`multiprocessing.JoinableQueue` into a single
interface that behaves identically on both :mod:`threading` and :mod:`multiprocessing` backends. As an example:

.. code-block:: pycon

    >>> from freethreading import Queue, Worker
    >>> queue = Queue()
    >>>
    >>> def producer():
    ...     for value in range(3):
    ...         queue.put(value)
    ...     queue.put(None)  # Sentinel marks completion
    ...
    >>> def consumer():
    ...     while True:
    ...         item = queue.get()
    ...         if item is None:
    ...             queue.task_done()
    ...             break
    ...         print(f"Processing {item}")
    ...         queue.task_done()
    ...
    >>> producer_worker = Worker(target=producer, name="Producer")
    >>> consumer_worker = Worker(target=consumer, name="Consumer")
    >>> producer_worker.start()
    >>> consumer_worker.start()
    >>> queue.join()
    >>> producer_worker.join()
    >>> consumer_worker.join()
    Processing 0
    Processing 1
    Processing 2


``SimpleQueue``
^^^^^^^^^^^^^^^

Similarly, :class:`freethreading.SimpleQueue` wraps the unbounded, lightweight :class:`queue.SimpleQueue` and
:class:`multiprocessing.SimpleQueue` into a single interface that behaves identically on both backends. For example:

.. code-block:: pycon

    >>> from freethreading import SimpleQueue, Worker
    >>> queue = SimpleQueue()
    >>>
    >>> def fill_queue():
    ...     queue.put("hello")
    ...     queue.put("world")
    ...
    >>> worker = Worker(target=fill_queue)
    >>> worker.start()
    >>> queue.get()
    'hello'
    >>> queue.get()
    'world'
    >>> worker.join()
    >>> queue.empty()
    True


Synchronization Primitives
--------------------------

:mod:`freethreading` offers the synchronization primitives common to :mod:`threading` and :mod:`multiprocessing`,
enabling worker coordination and control of shared resources.

Locks and Reentrant Locks
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: pycon

   >>> from freethreading import Lock, RLock
   >>>
   >>> lock = Lock()
   >>> rlock = RLock()
   >>>
   >>> def critical(name):
   ...     with lock:
   ...         print(f"'{name}' acquired the lock")
   ...
   >>> def recursive(n):
   ...     with rlock:
   ...         if n > 0:
   ...             recursive(n - 1)
   ...
   >>> critical("A")
   'A' acquired the lock
   >>> recursive(3)

Semaphores and Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: pycon

   >>> from freethreading import Semaphore, BoundedSemaphore, Condition
   >>> slots = Semaphore(2)  # limit concurrent producers
   >>> permits = BoundedSemaphore(3)  # prevents over-releasing
   >>> ready = Condition()
   >>> shared = []
   >>>
   >>> def producer(item):
   ...     with slots:  # only two producers execute at once
   ...         permits.acquire()
   ...         with ready:
   ...             shared.append(item)
   ...             ready.notify()
   ...     print(f"Produced '{item}'")
   ...
   >>> def consumer():
   ...     with ready:
   ...         while not shared:
   ...             ready.wait()
   ...         item = shared.pop()
   ...     print(f"Consumed '{item}'")
   ...     permits.release()  # bounded semaphore guarantees no double-release
   ...
   >>> producer("item-1")
   Produced 'item-1'
   >>> consumer()
   Consumed 'item-1'


Events and Barriers
^^^^^^^^^^^^^^^^^^^

Events let workers signal one another without busy-waiting:

.. code-block:: pycon

   >>> from freethreading import Event, Worker
   >>> event = Event()
   >>>
   >>> def wait_for_event():
   ...     print("Waiting for signal")
   ...     event.wait()
   ...     print("Signal received")
   ...
   >>> def send_event():
   ...     event.set()
   ...     print("Signal sent")
   ...
   >>> listener = Worker(target=wait_for_event, name="Listener")
   >>> sender = Worker(target=send_event, name="Sender")
   >>> listener.start()
   Waiting for signal
   >>> sender.start()
   Signal sent
   Signal received
   >>> listener.join()
   >>> sender.join()

Barriers bring a fixed number of workers to a common rendezvous point:

.. code-block:: pycon

   >>> from freethreading import Barrier, Worker
   >>> barrier = Barrier(3)
   >>>
   >>> def rendezvous(i):
   ...     print(f"Worker {i} ready")
   ...     barrier.wait()
   ...     print(f"Worker {i} proceeding")
   ...
   >>> workers = []
   >>> for i in range(3):
   ...     worker = Worker(target=rendezvous, args=(i,), name=f"worker-{i}")
   ...     worker.start()
   ...     workers.append(worker)
   ...
   Worker 0 ready
   Worker 1 ready
   Worker 2 ready
   Worker 2 proceeding
   Worker 0 proceeding
   Worker 1 proceeding
   >>> for worker in workers:
   ...     worker.join()


Inspection Utilities
--------------------

Query the runtime to aid debugging and observability.

.. code-block:: pycon

   >>> from freethreading import active_count, current_worker, enumerate, get_ident
   >>> from os import cpu_count
   >>>
   >>> cpu_count()
   8
   >>> active_count()
   1
   >>> current_worker().name
   'MainThread'
   >>> get_ident()  # Thread or process identifier
   123456
   >>> [worker.name for worker in enumerate()]
   ['MainThread']

End-to-End Example: Parallel Primes
-----------------------------------

The following example uses workers and queues to find primes in parallel without worrying
about the underlying backend.

.. code-block:: pycon

   >>> from os import cpu_count
   >>> from freethreading import Queue, Worker
   >>>
   >>> def is_prime(n):
   ...     if n < 2:
   ...         return False
   ...     for i in range(2, int(n ** 0.5) + 1):
   ...         if n % i == 0:
   ...             return False
   ...     return True
   ...
   >>> tasks = Queue()
   >>> results = Queue()
   >>>
   >>> def worker():
   ...     while True:
   ...         number = tasks.get()
   ...         if number is None:
   ...             tasks.task_done()
   ...             break
   ...         if is_prime(number):
   ...             results.put(number)
   ...         tasks.task_done()
   ...
   >>> pool = [Worker(target=worker) for _ in range(cpu_count())]
   >>> for w in pool:
   ...     w.start()
   ...
   >>> for candidate in range(10_000, 10_100):
   ...     tasks.put(candidate)
   ...
   >>> for _ in pool:
   ...     tasks.put(None)
   ...
   >>> tasks.join()
   >>> for w in pool:
   ...     w.join()
   ...
   >>> primes = []
   >>> while not results.empty():
   ...     primes.append(results.get())
   ...
   >>> print(f"Found {len(primes)} primes between 10_000 and 10_099.")
   Found 11 primes between 10_000 and 10_099.

This workflow delivers consistent behavior and true parallel execution out of the box,
regardless of whether your Python build supports free-threading.
