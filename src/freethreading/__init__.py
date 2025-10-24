"""
Unified concurrency API for portable parallel programming.

This module automatically selects between :mod:`threading` (when GIL is disabled)
and :mod:`multiprocessing` (when GIL is enabled) to provide optimal performance
across different Python builds.

The API exposes only the common subset of features available in both backends,
ensuring your code works identically regardless of the underlying implementation.

Examples
--------
Basic usage with automatic backend selection:

>>> import freethreading
>>> freethreading.get_backend()  # doctest: +SKIP
'multiprocessing'  # or 'threading' depending on Python build

Create workers that adapt to the backend:

>>> from freethreading import Worker, Queue
>>> def worker_func(q):
...     q.put("Hello from worker!")
>>> q = Queue()
>>> w = Worker(target=worker_func, args=(q,))
>>> w.start()
>>> w.join()
>>> q.get()
'Hello from worker!'

See Also
--------
threading : Threading-based parallelism
multiprocessing : Process-based parallelism
concurrent.futures : High-level interface for asynchronous execution

Notes
-----
Backend selection happens at import time by checking :func:`sys._is_gil_enabled`.
The chosen backend remains consistent throughout the program's lifetime.

All data passed to workers must be picklable when using the multiprocessing backend.
Use module-level functions instead of lambdas or nested functions.
"""

import pickle
import sys
import warnings
from typing import Literal

if sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True:
    from concurrent.futures import ProcessPoolExecutor as _PoolExecutor
    from multiprocessing import Barrier as _Barrier
    from multiprocessing import BoundedSemaphore as _BoundedSemaphore
    from multiprocessing import Condition as _Condition
    from multiprocessing import Event as _Event
    from multiprocessing import JoinableQueue as _Queue
    from multiprocessing import Lock as _Lock
    from multiprocessing import Process as _Worker
    from multiprocessing import RLock as _RLock
    from multiprocessing import Semaphore as _Semaphore
    from multiprocessing import SimpleQueue as _SimpleQueue
    from multiprocessing import active_children as _enumerate
    from multiprocessing import current_process as _current_worker
    from os import getpid as get_ident

    _backend = "multiprocessing"
else:
    from concurrent.futures import ThreadPoolExecutor as _PoolExecutor
    from queue import Queue as _Queue
    from queue import SimpleQueue as _SimpleQueue
    from threading import Barrier as _Barrier
    from threading import BoundedSemaphore as _BoundedSemaphore
    from threading import Condition as _Condition
    from threading import Event as _Event
    from threading import Lock as _Lock
    from threading import RLock as _RLock
    from threading import Semaphore as _Semaphore
    from threading import Thread as _Worker
    from threading import active_count as _active_count
    from threading import current_thread as _current_worker
    from threading import enumerate as _enumerate
    from threading import get_ident as _get_ident

    _backend = "threading"


class Barrier:
    """
    Synchronization barrier for coordinating :class:`Worker` objects.

    A barrier is used to wait for a fixed number of workers to reach a
    common point. Uses :class:`threading.Barrier` or
    :class:`multiprocessing.Barrier` depending on backend.

    Parameters
    ----------
    parties : int
        Number of workers required to pass the barrier.
    action : callable, optional
        Function called by one worker when the barrier is passed.
    timeout : float, optional
        Default timeout for :meth:`wait` calls.

    See Also
    --------
    Event : Event signaling between workers
    threading.Barrier : Threading implementation
    multiprocessing.Barrier : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Barrier, Worker
    >>> barrier = Barrier(3)  # Wait for 3 workers
    >>> def synchronized_task(i):
    ...     print(f"Worker {i} reached barrier")
    ...     barrier.wait()
    ...     print(f"Worker {i} past barrier")
    >>> workers = [Worker(target=synchronized_task, args=(i,)) for i in range(3)]
    >>> for w in workers:  # doctest: +SKIP
    ...     w.start()
    >>> for w in workers:  # doctest: +SKIP
    ...     w.join()
    """

    def __init__(self, parties, action=None, timeout=None):
        self._barrier = _Barrier(parties, action, timeout)

    def wait(self, timeout=None):
        """
        Wait until all parties have reached the barrier.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        int
            The arrival index (0 to parties-1).
        """
        return self._barrier.wait(timeout)

    def reset(self):
        """Reset the barrier to its initial empty state."""
        self._barrier.reset()

    def abort(self):
        """Put the barrier into a broken state."""
        self._barrier.abort()

    @property
    def parties(self):
        """
        The number of workers required to pass the barrier.

        Returns
        -------
        int
            Number of parties.
        """
        return self._barrier.parties

    @property
    def n_waiting(self):
        """
        The number of workers currently waiting at the barrier.

        Returns
        -------
        int
            Number of waiting workers.
        """
        return self._barrier.n_waiting

    @property
    def broken(self):
        """
        True if the barrier is in a broken state.

        Returns
        -------
        bool
            True if broken, False otherwise.
        """
        return self._barrier.broken


class BoundedSemaphore:
    """
    Unified BoundedSemaphore interface.

    A bounded semaphore that prevents the counter from exceeding its initial value.
    Uses :class:`threading.BoundedSemaphore` or
    :class:`multiprocessing.BoundedSemaphore` depending on backend.

    Parameters
    ----------
    value : int, default=1
        Initial value for the semaphore counter.

    See Also
    --------
    Semaphore : Standard semaphore without upper bound
    threading.BoundedSemaphore : Threading implementation
    multiprocessing.BoundedSemaphore : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import BoundedSemaphore
    >>> sem = BoundedSemaphore(2)
    >>> sem.acquire()
    True
    >>> sem.acquire()
    True
    >>> sem.release()
    >>> sem.release()
    """

    def __init__(self, value=1):
        self._semaphore = _BoundedSemaphore(value)

    def acquire(self, blocking=True, timeout=None):
        """
        Acquire the semaphore.

        Parameters
        ----------
        blocking : bool, default=True
            If True, block until the semaphore can be acquired.
        timeout : float, optional
            Maximum time to wait in seconds. None means wait forever.

        Returns
        -------
        bool
            True if acquired, False if timeout occurred.
        """
        return self._semaphore.acquire(blocking, timeout)

    def release(self):
        """Release the semaphore, incrementing the counter."""
        self._semaphore.release()

    def __enter__(self):
        """Enter the runtime context (acquire the semaphore)."""
        return self._semaphore.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (release the semaphore)."""
        return self._semaphore.__exit__(exc_type, exc_val, exc_tb)


class Condition:
    """
    Unified Condition variable interface.

    Allows workers to wait for a specific condition to become true.
    Uses :class:`threading.Condition` or :class:`multiprocessing.Condition`.

    Parameters
    ----------
    lock : Lock or RLock, optional
        Underlying lock to use. If not provided, a new :class:`RLock` is created.

    See Also
    --------
    Lock : Mutual exclusion lock
    RLock : Reentrant lock
    threading.Condition : Threading implementation
    multiprocessing.Condition : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Condition, Worker
    >>> condition = Condition()
    >>> items = []
    >>> def producer():
    ...     with condition:
    ...         items.append("item")
    ...         condition.notify()
    >>> def consumer():
    ...     with condition:
    ...         condition.wait()
    ...         item = items.pop() if items else None
    """

    def __init__(self, lock=None):
        self._condition = _Condition(lock)

    def acquire(self, blocking=True, timeout=None):
        """
        Acquire the underlying lock.

        Parameters
        ----------
        blocking : bool, default=True
            If True, block until the lock can be acquired.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if acquired, False if timeout occurred.
        """
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._condition.acquire(blocking, timeout)

    def release(self):
        """Release the underlying lock."""
        self._condition.release()

    def wait(self, timeout=None):
        """
        Wait until notified or timeout occurs.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True unless timeout occurred.
        """
        return self._condition.wait(timeout)

    def wait_for(self, predicate, timeout=None):
        """
        Wait until a predicate becomes true.

        Parameters
        ----------
        predicate : callable
            Function that returns a boolean value.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            The predicate result.
        """
        return self._condition.wait_for(predicate, timeout)

    def notify(self, n=1):
        """
        Wake up one or more workers waiting on this condition.

        Parameters
        ----------
        n : int, default=1
            Number of workers to wake up.
        """
        self._condition.notify(n)

    def notify_all(self):
        """Wake up all workers waiting on this condition."""
        self._condition.notify_all()

    def __enter__(self):
        """Enter the runtime context (acquire the lock)."""
        return self._condition.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (release the lock)."""
        return self._condition.__exit__(exc_type, exc_val, exc_tb)


class Event:
    """
    Synchronization primitive for signaling between :class:`Worker` objects.

    An event manages an internal flag that can be set or cleared. Workers can
    wait for the flag to be set. Uses :class:`threading.Event` or
    :class:`multiprocessing.Event` depending on backend.

    See Also
    --------
    Barrier : Synchronization barrier for coordinating workers
    threading.Event : Threading implementation
    multiprocessing.Event : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Event, Worker
    >>> event = Event()
    >>> def waiter():
    ...     event.wait()
    ...     print("Event set!")
    >>> def setter():
    ...     import time
    ...     time.sleep(0.1)
    ...     event.set()
    >>> w1 = Worker(target=waiter)
    >>> w2 = Worker(target=setter)
    >>> w1.start()
    >>> w2.start()
    >>> w1.join()  # doctest: +SKIP
    >>> w2.join()  # doctest: +SKIP
    Event set!
    """

    def __init__(self):
        self._event = _Event()

    def is_set(self):
        """
        Return True if the internal flag is set.

        Returns
        -------
        bool
            True if set, False otherwise.
        """
        return self._event.is_set()

    def set(self):
        """Set the internal flag, waking up all waiting workers."""
        self._event.set()

    def clear(self):
        """Reset the internal flag to false."""
        self._event.clear()

    def wait(self, timeout=None):
        """
        Block until the internal flag is true.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if flag is set, False if timeout occurred.
        """
        return self._event.wait(timeout)


class Lock:
    """
    Unified Lock interface.

    A mutual exclusion lock. Uses :class:`threading.Lock` or
    :class:`multiprocessing.Lock` depending on backend.

    See Also
    --------
    RLock : Reentrant lock that can be acquired multiple times by the same worker
    threading.Lock : Threading implementation
    multiprocessing.Lock : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Lock
    >>> lock = Lock()
    >>> with lock:
    ...     # Critical section - only one worker at a time
    ...     pass
    """

    def __init__(self):
        self._lock = _Lock()

    def acquire(self, blocking=True, timeout=None):
        """
        Acquire the lock.

        Parameters
        ----------
        blocking : bool, default=True
            If True, block until the lock can be acquired.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if acquired, False if timeout occurred.
        """
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._lock.acquire(blocking, timeout)

    def release(self):
        """Release the lock."""
        self._lock.release()

    def locked(self):
        """
        Return True if the lock is currently held.

        Returns
        -------
        bool
            True if locked, False otherwise.
        """
        if hasattr(self._lock, "locked"):
            return self._lock.locked()

        # Fallback for Python < 3.14
        if self.acquire(blocking=False):
            self.release()
            return False
        return True

    def __enter__(self):
        """Enter the runtime context (acquire the lock)."""
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (release the lock)."""
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class Queue:
    """
    Unified Queue interface for worker communication.

    A FIFO queue supporting task tracking with :meth:`task_done` and :meth:`join`.
    Uses :class:`queue.Queue` or :class:`multiprocessing.JoinableQueue`.

    Parameters
    ----------
    maxsize : int, default=0
        Maximum number of items allowed in the queue. 0 means unlimited.

    See Also
    --------
    SimpleQueue : Simpler queue without task tracking
    queue.Queue : Threading implementation
    multiprocessing.JoinableQueue : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Queue, Worker
    >>> q = Queue()
    >>> def worker(q):
    ...     while True:
    ...         item = q.get()
    ...         if item is None:
    ...             q.task_done()
    ...             break
    ...         # Process item
    ...         q.task_done()
    >>> q.put("item1")
    >>> q.put(None)  # Sentinel
    >>> w = Worker(target=worker, args=(q,))
    >>> w.start()
    >>> q.join()  # Wait for all tasks
    >>> w.join()
    """

    def __init__(self, maxsize=0):
        self._queue = _Queue(maxsize)

    def put(self, item, block=True, timeout=None):
        """
        Put an item into the queue.

        Parameters
        ----------
        item
            Item to add to the queue.
        block : bool, default=True
            If True, block until space is available.
        timeout : float, optional
            Maximum time to wait in seconds.
        """
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.

        Parameters
        ----------
        block : bool, default=True
            If True, block until an item is available.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        item
            The next item from the queue.
        """
        return self._queue.get(block=block, timeout=timeout)

    def task_done(self):
        """Indicate that a formerly enqueued task is complete."""
        self._queue.task_done()

    def join(self):
        """Block until all items have been gotten and processed."""
        self._queue.join()

    def qsize(self):
        """
        Return the approximate size of the queue.

        Returns
        -------
        int
            Number of items in the queue.
        """
        return self._queue.qsize()

    def empty(self):
        """
        Return True if the queue is empty.

        Returns
        -------
        bool
            True if empty, False otherwise.
        """
        return self._queue.empty()

    def full(self):
        """
        Return True if the queue is full.

        Returns
        -------
        bool
            True if full, False otherwise.
        """
        return self._queue.full()

    def put_nowait(self, item):
        """
        Put an item into the queue without blocking.

        Parameters
        ----------
        item
            Item to add to the queue.

        Raises
        ------
        queue.Full
            If queue is full.
        """
        self._queue.put_nowait(item)

    def get_nowait(self):
        """
        Remove and return an item without blocking.

        Returns
        -------
        item
            The next item from the queue.

        Raises
        ------
        queue.Empty
            If queue is empty.
        """
        return self._queue.get_nowait()


class RLock:
    """
    Unified RLock (reentrant lock) interface.

    A lock that can be acquired multiple times by the same worker.
    Uses :class:`threading.RLock` or :class:`multiprocessing.RLock`.

    See Also
    --------
    Lock : Standard mutual exclusion lock
    threading.RLock : Threading implementation
    multiprocessing.RLock : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import RLock
    >>> rlock = RLock()
    >>> def recursive_function(n):
    ...     with rlock:
    ...         if n > 0:
    ...             recursive_function(n - 1)
    >>> recursive_function(5)
    """

    def __init__(self):
        self._lock = _RLock()

    def acquire(self, blocking=True, timeout=None):
        """
        Acquire the lock, incrementing the recursion level.

        Parameters
        ----------
        blocking : bool, default=True
            If True, block until the lock can be acquired.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if acquired, False if timeout occurred.
        """
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._lock.acquire(blocking, timeout)

    def release(self):
        """Release the lock, decrementing the recursion level."""
        self._lock.release()

    def __enter__(self):
        """Enter the runtime context (acquire the lock)."""
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (release the lock)."""
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class Semaphore:
    """
    Unified Semaphore interface.

    A semaphore manages an internal counter decremented by :meth:`acquire` calls
    and incremented by :meth:`release` calls. Uses :class:`threading.Semaphore`
    or :class:`multiprocessing.Semaphore`.

    Parameters
    ----------
    value : int, default=1
        Initial value for the semaphore counter.

    See Also
    --------
    BoundedSemaphore : Semaphore with upper bound protection
    threading.Semaphore : Threading implementation
    multiprocessing.Semaphore : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import Semaphore, Worker
    >>> sem = Semaphore(3)  # Allow 3 concurrent accesses
    >>> def limited_resource():
    ...     with sem:
    ...         # Only 3 workers can be here at once
    ...         pass
    """

    def __init__(self, value=1):
        self._semaphore = _Semaphore(value)

    def acquire(self, blocking=True, timeout=None):
        """
        Acquire the semaphore.

        Parameters
        ----------
        blocking : bool, default=True
            If True, block until the semaphore can be acquired.
        timeout : float, optional
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if acquired, False if timeout occurred.
        """
        return self._semaphore.acquire(blocking, timeout)

    def release(self):
        """Release the semaphore, incrementing the counter."""
        self._semaphore.release()

    def __enter__(self):
        """Enter the runtime context (acquire the semaphore)."""
        return self._semaphore.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (release the semaphore)."""
        return self._semaphore.__exit__(exc_type, exc_val, exc_tb)


class SimpleQueue:
    """
    Unified SimpleQueue interface.

    A simpler FIFO queue without task tracking or size limits.
    Uses :class:`queue.SimpleQueue` or :class:`multiprocessing.SimpleQueue`.

    .. warning::
       The multiprocessing backend does not support ``block`` and ``timeout``
       parameters. Using them will issue a runtime warning and ignore the parameters.

    See Also
    --------
    Queue : Full-featured queue with task tracking
    queue.SimpleQueue : Threading implementation
    multiprocessing.SimpleQueue : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import SimpleQueue
    >>> sq = SimpleQueue()
    >>> sq.put("Hello")
    >>> sq.get()
    'Hello'
    >>> sq.empty()
    True
    """

    def __init__(self):
        self._queue = _SimpleQueue()

    def put(self, item, block=True, timeout=None):
        """
        Put an item into the queue.

        Parameters
        ----------
        item
            Item to add to the queue.
        block : bool, default=True
            If True, block until space is available (threading only).
        timeout : float, optional
            Maximum time to wait in seconds (threading only).

        Warnings
        --------
        The multiprocessing backend ignores ``block`` and ``timeout`` parameters.
        """
        if _backend == "threading":
            self._queue.put(item, block=block, timeout=timeout)
        else:
            if not block or timeout is not None:
                warnings.warn(
                    "SimpleQueue with multiprocessing backend does not support "
                    "block or timeout parameters. Arguments ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._queue.put(item)

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.

        Parameters
        ----------
        block : bool, default=True
            If True, block until an item is available (threading only).
        timeout : float, optional
            Maximum time to wait in seconds (threading only).

        Returns
        -------
        item
            The next item from the queue.

        Warnings
        --------
        The multiprocessing backend ignores ``block`` and ``timeout`` parameters
        and always blocks until an item is available.
        """
        if _backend == "threading":
            return self._queue.get(block=block, timeout=timeout)
        else:
            if not block or timeout is not None:
                warnings.warn(
                    "SimpleQueue with multiprocessing backend does not support "
                    "block or timeout parameters. Will block until item available.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._queue.get()

    def empty(self):
        """
        Return True if the queue is empty.

        Returns
        -------
        bool
            True if empty, False otherwise.
        """
        return self._queue.empty()


class Worker:
    """
    Unified Worker interface (Thread or Process).

    Represents an activity that runs in a separate thread or process.
    Uses :class:`threading.Thread` or :class:`multiprocessing.Process`.

    Parameters
    ----------
    group : None
        Reserved for future extension (always None).
    target : callable, optional
        Function to be invoked by the :meth:`run` method.
    name : str, optional
        Worker name for identification.
    args : tuple, default=()
        Positional arguments for the target function.
    kwargs : dict, optional
        Keyword arguments for the target function.
    daemon : bool, optional
        Whether the worker is a daemon. Daemon workers are terminated when
        the program exits.

    .. warning::
       All arguments passed to workers must be picklable when using the
       multiprocessing backend. Avoid lambdas and nested functions.

    See Also
    --------
    threading.Thread : Threading implementation
    multiprocessing.Process : Multiprocessing implementation
    WorkerPoolExecutor : High-level interface for managing worker pools

    Examples
    --------
    >>> from freethreading import Worker
    >>> def task(name):
    ...     print(f"Hello from {name}")
    >>> w = Worker(target=task, args=("Worker-1",))
    >>> w.start()
    >>> w.join()  # doctest: +SKIP
    Hello from Worker-1
    """

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        if kwargs is None:
            kwargs = {}

        try:
            pickle.dumps((target, args, kwargs))
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            warnings.warn(
                f"Arguments may not be picklable: {e}. "
                f"This will cause issues with multiprocessing backend. "
                f"Use module-level functions instead of lambdas.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._worker = _Worker(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )

    def start(self):
        """Start the worker's activity."""
        self._worker.start()

    def join(self, timeout=None):
        """
        Wait for the worker to terminate.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.
        """
        self._worker.join(timeout)

    def is_alive(self):
        """
        Return whether the worker is alive.

        Returns
        -------
        bool
            True if the worker is still running.
        """
        return self._worker.is_alive()

    @property
    def name(self):
        """
        Worker name.

        Returns
        -------
        str
            The worker's name.
        """
        return self._worker.name

    @name.setter
    def name(self, value):
        """Set the worker name."""
        self._worker.name = value

    @property
    def daemon(self):
        """
        Daemon status.

        Returns
        -------
        bool
            True if the worker is a daemon.
        """
        return self._worker.daemon

    @daemon.setter
    def daemon(self, value):
        """Set the daemon status."""
        self._worker.daemon = value


class WorkerPoolExecutor:
    """
    Executor that manages a pool of :class:`Worker` objects.

    Provides a high-level interface for asynchronously executing callables using
    a pool of workers. Uses :class:`concurrent.futures.ThreadPoolExecutor` or
    :class:`concurrent.futures.ProcessPoolExecutor` depending on backend.

    Parameters
    ----------
    max_workers : int, optional
        Maximum number of workers in the pool.
    **kwargs
        Additional arguments passed to the underlying executor.

    See Also
    --------
    Worker : Lower-level worker interface
    concurrent.futures.ThreadPoolExecutor : Threading implementation
    concurrent.futures.ProcessPoolExecutor : Multiprocessing implementation

    Examples
    --------
    >>> from freethreading import WorkerPoolExecutor
    >>> def square(x):
    ...     return x * x
    >>> with WorkerPoolExecutor(max_workers=4) as executor:
    ...     results = list(executor.map(square, range(10)))
    >>> results
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """

    def __init__(self, max_workers=None, **kwargs):
        self._executor = _PoolExecutor(max_workers=max_workers, **kwargs)

    def submit(self, fn, *args, **kwargs):
        """
        Submit a callable to be executed.

        Parameters
        ----------
        fn : callable
            The callable to execute.
        *args
            Positional arguments for fn.
        **kwargs
            Keyword arguments for fn.

        Returns
        -------
        Future
            A Future representing the execution.
        """
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """
        Map a function over iterables.

        Parameters
        ----------
        fn : callable
            Function to apply.
        *iterables
            Iterables to map over.
        timeout : float, optional
            Maximum time to wait for results.
        chunksize : int, default=1
            Size of chunks for multiprocessing.

        Returns
        -------
        iterator
            Iterator over results.
        """
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait=True, cancel_futures=False):
        """
        Shutdown the executor.

        Parameters
        ----------
        wait : bool, default=True
            If True, wait for pending futures to complete.
        cancel_futures : bool, default=False
            If True, cancel pending futures.
        """
        return self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        """Enter the runtime context."""
        return self._executor.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context (shutdown the executor)."""
        return self._executor.__exit__(exc_type, exc_val, exc_tb)


def active_count():
    """
    Return the number of currently active :class:`Worker` objects.

    Returns
    -------
    int
        Number of :class:`Worker` objects currently running.

    See Also
    --------
    enumerate : Return a list of all active :class:`Worker` objects
    current_worker : Get the current :class:`Worker` object

    Notes
    -----
    This counts all :class:`Worker` objects (threads or processes) that have
    been started but not yet finished.

    Examples
    --------
    >>> from freethreading import active_count, Worker
    >>> def task():
    ...     pass
    >>> initial = active_count()
    >>> w = Worker(target=task)
    >>> w.start()
    >>> active_count() >= initial + 1  # doctest: +SKIP
    True
    """
    return _active_count() if _backend == "threading" else len(enumerate())


def current_worker():
    """
    Return the current worker object.

    Returns
    -------
    Thread | BaseProcess
        The underlying thread or process object corresponding to the caller.
        Provides common attributes: ``name``, ``daemon``, ``ident``,
        and methods: ``is_alive()``, ``join()``, ``start()``, ``run()``.

    See Also
    --------
    get_ident : Get the identifier of the current worker
    active_count : Get the number of active workers

    Notes
    -----
    Backend-specific attributes like ``pid`` (processes) or ``native_id`` (threads)
    are also available but not portable across backends.

    Examples
    --------
    >>> from freethreading import current_worker
    >>> worker = current_worker()
    >>> print(worker.name)  # doctest: +SKIP
    MainThread
    """
    return _current_worker()


def enumerate():
    """
    Return a list of all active worker objects.

    Returns
    -------
    list of Thread | BaseProcess
        List of all underlying thread or process objects currently alive.
        Each object provides common attributes: ``name``, ``daemon``, ``ident``,
        and methods: ``is_alive()``, ``join()``, ``start()``, ``run()``.

    See Also
    --------
    active_count : Get the count of active workers
    current_worker : Get the current worker

    Notes
    -----
    Backend-specific attributes like ``pid`` (processes) or ``native_id`` (threads)
    are also available but not portable across backends.

    Examples
    --------
    >>> from freethreading import enumerate, Worker
    >>> def task():
    ...     pass
    >>> w = Worker(target=task)
    >>> w.start()
    >>> workers = enumerate()
    >>> len(workers) >= 2  # At least main + our worker  # doctest: +SKIP
    True
    """
    return _enumerate()


def get_backend() -> Literal["threading", "multiprocessing"]:
    """
    Get the name of the active concurrency backend.

    Returns
    -------
    Literal['threading', 'multiprocessing']
        'threading' when GIL is disabled, 'multiprocessing' when GIL is enabled.

    See Also
    --------
    is_gil_enabled : Check if GIL is enabled

    Examples
    --------
    >>> import freethreading
    >>> backend = freethreading.get_backend()  # doctest: +SKIP
    >>> print(f"Using {backend} backend")  # doctest: +SKIP
    Using multiprocessing backend
    """
    return _backend


def get_ident():
    """
    Return the identifier of the current worker.

    Returns
    -------
    int
        Thread identifier or process ID of the current :class:`Worker`.

    See Also
    --------
    current_worker : Get the current Worker object

    Notes
    -----
    - When using threading backend: Returns thread identifier
    - When using multiprocessing backend: Returns process ID (PID)

    Examples
    --------
    >>> from freethreading import get_ident
    >>> ident = get_ident()
    >>> isinstance(ident, int)
    True
    """
    return _get_ident()


__all__ = [
    "Barrier",
    "BoundedSemaphore",
    "Condition",
    "Event",
    "Lock",
    "Queue",
    "RLock",
    "Semaphore",
    "SimpleQueue",
    "Worker",
    "WorkerPoolExecutor",
    "active_count",
    "current_worker",
    "enumerate",
    "get_backend",
    "get_ident",
]
