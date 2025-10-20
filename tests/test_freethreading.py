"""Tests for freethreading module on both threading and multiprocessing backends."""

import importlib
import sys
import warnings

import pytest


# Module-level functions for multiprocessing picklability
def simple_task():
    pass


def task_with_args(x, y):
    return x + y


def task_with_kwargs(x, y=10):
    return x + y


def square(x):
    return x * x


def task_that_returns_value():
    return 42


@pytest.fixture(params=[True, False], ids=["multiprocessing", "threading"])
def backend(request, monkeypatch):
    """Fixture that tests both backends by mocking GIL status."""
    gil_enabled = request.param

    if gil_enabled:
        monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
    else:
        monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    import freethreading

    importlib.reload(freethreading)

    yield freethreading

    importlib.reload(freethreading)


def test_is_gil_enabled(backend):
    result = backend.is_gil_enabled()
    assert isinstance(result, bool)


def test_get_backend(backend):
    result = backend.get_backend()
    assert result in ["threading", "multiprocessing"]


def test_backend_consistency(backend):
    if backend.is_gil_enabled():
        assert backend.get_backend() == "multiprocessing"
    else:
        assert backend.get_backend() == "threading"


def test_cpu_count(backend):
    count = backend.cpu_count()
    assert isinstance(count, int)
    assert count > 0


def test_get_ident(backend):
    ident = backend.get_ident()
    assert isinstance(ident, int)
    assert ident > 0


def test_current_worker(backend):
    worker = backend.current_worker()
    assert worker is not None


def test_active_count(backend):
    count = backend.active_count()
    assert isinstance(count, int)
    assert count >= 0


def test_enumerate(backend):
    workers = backend.enumerate()
    assert isinstance(workers, list)


def test_worker_creation(backend):
    worker = backend.Worker(target=simple_task)
    assert worker is not None
    worker.start()
    worker.join()


def test_worker_with_args(backend):
    worker = backend.Worker(target=task_with_args, args=(1, 2))
    worker.start()
    worker.join()


def test_worker_with_kwargs(backend):
    worker = backend.Worker(target=task_with_kwargs, kwargs={"x": 5, "y": 15})
    worker.start()
    worker.join()


def test_worker_with_name(backend):
    worker = backend.Worker(target=simple_task, name="TestWorker")
    assert worker.name == "TestWorker"
    worker.start()
    worker.join()


def test_worker_daemon(backend):
    worker = backend.Worker(target=simple_task, daemon=True)
    assert worker.daemon is True
    worker.start()
    worker.join()


def test_worker_pickling_warning(backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = backend.Worker(target=lambda: None)
        assert len(w) == 1
        assert "not be picklable" in str(w[0].message)


def test_worker_local_function_warning(backend):
    def local_function():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = backend.Worker(target=local_function)
        assert len(w) == 1
        assert "not be picklable" in str(w[0].message)


def test_lock(backend):
    lock = backend.Lock()
    lock.acquire()
    lock.release()


def test_lock_locked(backend):
    lock = backend.Lock()
    assert not lock.locked()

    lock.acquire()
    assert lock.locked()

    lock.release()
    assert not lock.locked()


def test_lock_context_manager(backend):
    lock = backend.Lock()
    assert not lock.locked()

    with lock:
        assert lock.locked()

    assert not lock.locked()


def test_lock_acquire_timeout(backend):
    lock = backend.Lock()

    acquired = lock.acquire(timeout=0.1)
    assert acquired

    acquired2 = lock.acquire(blocking=False)
    assert not acquired2

    lock.release()


def test_rlock(backend):
    lock = backend.RLock()
    lock.acquire()
    lock.acquire()
    lock.release()
    lock.release()


def test_rlock_acquire_release(backend):
    """Test RLock basic acquire/release."""
    lock = backend.RLock()
    assert lock.acquire()
    assert lock.acquire()
    lock.release()
    lock.release()


def test_rlock_context_manager(backend):
    lock = backend.RLock()

    with lock:
        with lock:
            pass


def test_semaphore(backend):
    sem = backend.Semaphore(2)
    sem.acquire()
    sem.release()


def test_semaphore_context_manager(backend):
    sem = backend.Semaphore(1)
    with sem:
        pass


def test_semaphore_context_manager_exception(backend):
    sem = backend.Semaphore(1)
    try:
        with sem:
            raise ValueError("test")
    except ValueError:
        pass
    assert not sem.acquire(blocking=False) or sem.release() is None


def test_semaphore_timeout(backend):
    sem = backend.Semaphore(1)

    acquired = sem.acquire(timeout=0.1)
    assert acquired

    acquired2 = sem.acquire(blocking=False)
    assert not acquired2

    sem.release()


def test_bounded_semaphore(backend):
    sem = backend.BoundedSemaphore(2)
    sem.acquire()
    sem.release()


def test_bounded_semaphore_context_manager_exception(backend):
    sem = backend.BoundedSemaphore(1)
    try:
        with sem:
            raise ValueError("test")
    except ValueError:
        pass
    assert not sem.acquire(blocking=False) or sem.release() is None


def test_event(backend):
    event = backend.Event()
    assert not event.is_set()

    event.set()
    assert event.is_set()

    event.clear()
    assert not event.is_set()


def test_condition(backend):
    cond = backend.Condition()
    acquired = cond.acquire()
    assert acquired is True
    cond.release()


def test_condition_context_manager(backend):
    cond = backend.Condition()
    with cond:
        assert cond is not None


def test_condition_context_manager_exception(backend):
    cond = backend.Condition()
    exception_raised = False
    try:
        with cond:
            raise ValueError("test")
    except ValueError:
        exception_raised = True
    assert exception_raised
    assert cond.acquire()
    cond.release()


def test_condition_wait_for(backend):
    cond = backend.Condition()
    flag = []

    def predicate():
        return len(flag) > 0

    flag.append(1)

    with cond:
        result = cond.wait_for(predicate, timeout=0.1)
        assert result


def test_condition_notify(backend):
    cond = backend.Condition()

    with cond:
        cond.notify(1)
        cond.notify_all()


def test_condition_notify_with_count(backend):
    """Test Condition.notify(n) with different count values."""
    cond = backend.Condition()
    assert cond.acquire()
    cond.notify(0)
    cond.notify(10)
    cond.release()
    assert cond.acquire()
    cond.release()


def test_barrier(backend):
    barrier = backend.Barrier(1)
    barrier.wait()


def test_queue(backend):
    q = backend.Queue()
    q.put(42)
    result = q.get()
    assert result == 42


def test_queue_multiple_items(backend):
    q = backend.Queue()

    for i in range(5):
        q.put(i)

    results = []
    for _ in range(5):
        results.append(q.get())

    assert results == [0, 1, 2, 3, 4]


def test_queue_nowait_methods(backend):
    import queue
    import time

    q = backend.Queue()

    q.put_nowait(42)

    if backend.get_backend() == "multiprocessing":
        time.sleep(0.01)

    result = q.get_nowait()
    assert result == 42

    try:
        q.get_nowait()
        assert False, "Should have raised Empty"
    except queue.Empty:
        pass


def test_queue_qsize(backend):
    """Test Queue.qsize() method."""
    q = backend.Queue()
    assert q.qsize() == 0
    q.put(1)
    assert q.qsize() == 1
    q.get()
    assert q.qsize() == 0


def test_queue_empty_full(backend):
    import time

    q = backend.Queue(maxsize=2)

    assert q.empty()
    assert not q.full()

    q.put(1)

    if backend.get_backend() == "multiprocessing":
        time.sleep(0.01)

    assert not q.empty()
    assert not q.full()

    q.put(2)

    if backend.get_backend() == "multiprocessing":
        time.sleep(0.01)

    assert not q.empty()
    assert q.full()

    q.get()
    q.get()
    assert q.empty()


def test_simple_queue(backend):
    q = backend.SimpleQueue()
    q.put(42)
    result = q.get()
    assert result == 42


def test_simple_queue_empty(backend):
    """Test SimpleQueue.empty() method."""
    q = backend.SimpleQueue()
    assert q.empty()
    q.put(1)
    assert not q.empty()
    q.get()
    assert q.empty()


def test_simple_queue_blocking_warnings(backend):
    q = backend.SimpleQueue()

    if backend.get_backend() == "multiprocessing":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q.put(1, block=False)
            assert len(w) == 1
            assert "does not support" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = q.get(timeout=0.1)
            assert len(w) == 1
            assert "does not support" in str(w[0].message)
            assert result == 1
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q.put(1, block=False)
            assert len(w) == 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q.put(2, timeout=0.1)
            assert len(w) == 0

        result1 = q.get(block=False)
        result2 = q.get(timeout=0.1)
        assert result1 == 1
        assert result2 == 2


def test_pool_executor(backend):
    with backend.PoolExecutor(max_workers=2) as executor:
        results = list(executor.map(square, [1, 2, 3, 4]))

    assert results == [1, 4, 9, 16]


def test_pool_executor_submit(backend):
    with backend.PoolExecutor(max_workers=2) as executor:
        future = executor.submit(square, 5)
        result = future.result()

    assert result == 25
