"""
Unified concurrency API that automatically selects threading (GIL disabled)
or multiprocessing (GIL enabled) based on your Python build.

Exposes only features available in both backends.
"""

import pickle
import sys
import warnings
from typing import Literal


def is_gil_enabled() -> bool:
    """Check if the GIL is enabled."""
    return sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True


def get_backend() -> Literal["threading", "multiprocessing"]:
    """Return 'threading' or 'multiprocessing'."""
    return _backend


if is_gil_enabled():
    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    from multiprocessing import (
        Barrier,
        Event,
        cpu_count,
    )
    from multiprocessing import BoundedSemaphore as _BoundedSemaphore
    from multiprocessing import Condition as _Condition
    from multiprocessing import JoinableQueue as _Queue
    from multiprocessing import Lock as _Lock
    from multiprocessing import Process as _Worker
    from multiprocessing import RLock as _RLock
    from multiprocessing import Semaphore as _Semaphore
    from multiprocessing import SimpleQueue as _SimpleQueue
    from multiprocessing import active_children as enumerate
    from multiprocessing import current_process as current_worker
    from os import getpid as get_ident

    def active_count():
        return len(enumerate())

    _backend = "multiprocessing"
else:
    from concurrent.futures import ThreadPoolExecutor as PoolExecutor
    from os import cpu_count
    from queue import Queue as _Queue
    from queue import SimpleQueue as _SimpleQueue
    from threading import (
        Barrier,
        Event,
        active_count,
        enumerate,
        get_ident,
    )
    from threading import BoundedSemaphore as _BoundedSemaphore
    from threading import Condition as _Condition
    from threading import Lock as _Lock
    from threading import RLock as _RLock
    from threading import Semaphore as _Semaphore
    from threading import Thread as _Worker
    from threading import current_thread as current_worker

    _backend = "threading"


class BoundedSemaphore:
    """Unified BoundedSemaphore interface."""

    def __init__(self, value=1):
        self._semaphore = _BoundedSemaphore(value)

    def acquire(self, blocking=True, timeout=None):
        return self._semaphore.acquire(blocking, timeout)

    def release(self):
        self._semaphore.release()

    def __enter__(self):
        return self._semaphore.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._semaphore.__exit__(exc_type, exc_val, exc_tb)


class Condition:
    """Unified Condition interface."""

    def __init__(self, lock=None):
        self._condition = _Condition(lock)

    def acquire(self, blocking=True, timeout=None):
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._condition.acquire(blocking, timeout)

    def release(self):
        self._condition.release()

    def wait(self, timeout=None):
        return self._condition.wait(timeout)

    def wait_for(self, predicate, timeout=None):
        return self._condition.wait_for(predicate, timeout)

    def notify(self, n=1):
        self._condition.notify(n)

    def notify_all(self):
        self._condition.notify_all()

    def __enter__(self):
        return self._condition.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._condition.__exit__(exc_type, exc_val, exc_tb)


class Lock:
    """Unified Lock interface."""

    def __init__(self):
        self._lock = _Lock()

    def acquire(self, blocking=True, timeout=None):
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        if hasattr(self._lock, "locked"):
            return self._lock.locked()

        # Fallback for Python < 3.14
        if self.acquire(blocking=False):
            self.release()
            return False
        return True

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class Queue:
    """Unified Queue interface."""

    def __init__(self, maxsize=0):
        self._queue = _Queue(maxsize)

    def put(self, item, block=True, timeout=None):
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        return self._queue.get(block=block, timeout=timeout)

    def task_done(self):
        """Indicate that a formerly enqueued task is complete."""
        self._queue.task_done()

    def join(self):
        """Block until all items have been gotten and processed."""
        self._queue.join()

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def put_nowait(self, item):
        self._queue.put_nowait(item)

    def get_nowait(self):
        return self._queue.get_nowait()


class RLock:
    """Unified RLock interface."""

    def __init__(self):
        self._lock = _RLock()

    def acquire(self, blocking=True, timeout=None):
        if timeout is None and _backend == "threading":
            timeout = -1
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        if hasattr(self._lock, "locked"):
            return self._lock.locked()

        # Fallback for Python < 3.14
        if self.acquire(blocking=False):
            self.release()
            return False
        return True

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class Semaphore:
    """Unified Semaphore interface."""

    def __init__(self, value=1):
        self._semaphore = _Semaphore(value)

    def acquire(self, blocking=True, timeout=None):
        return self._semaphore.acquire(blocking, timeout)

    def release(self):
        self._semaphore.release()

    def __enter__(self):
        return self._semaphore.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._semaphore.__exit__(exc_type, exc_val, exc_tb)


class SimpleQueue:
    """Unified SimpleQueue interface."""

    def __init__(self):
        self._queue = _SimpleQueue()

    def put(self, item, block=True, timeout=None):
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
        return self._queue.empty()


class Worker(_Worker):
    """Unified Worker interface (Thread or Process)."""

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

        super().__init__(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )


__all__ = [
    "Barrier",
    "BoundedSemaphore",
    "Condition",
    "Event",
    "Lock",
    "PoolExecutor",
    "Queue",
    "RLock",
    "Semaphore",
    "SimpleQueue",
    "Worker",
    "active_count",
    "cpu_count",
    "current_worker",
    "enumerate",
    "get_backend",
    "get_ident",
    "is_gil_enabled",
]
