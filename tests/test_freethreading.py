"""Tests for freethreading module on both threading and multiprocessing backends."""

import importlib
import sys
import threading
import time
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
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: request.param, raising=False)
    import freethreading

    return importlib.reload(freethreading)


def test_get_backend(backend):
    result = backend.get_backend()
    assert result in ["threading", "multiprocessing"]


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


def test_worker_with_args(backend):
    worker = backend.Worker(target=task_with_args, args=(1, 2))
    assert worker is not None
    worker.start()
    worker.join()
    if backend.get_backend() == "threading":
        import threading

        assert isinstance(worker._worker, threading.Thread)
    else:
        import multiprocessing

        assert isinstance(worker._worker, multiprocessing.Process)


def test_worker_with_kwargs(backend):
    worker = backend.Worker(target=task_with_kwargs, kwargs={"x": 5, "y": 15})
    assert worker is not None
    worker.start()
    worker.join()
    if backend.get_backend() == "threading":
        import threading

        assert isinstance(worker._worker, threading.Thread)
    else:
        import multiprocessing

        assert isinstance(worker._worker, multiprocessing.Process)


def test_worker_name_property(backend):
    worker = backend.Worker(target=simple_task, name="InitialName")
    assert worker.name == "InitialName"

    # Set new name before starting
    worker.name = "NewName"
    assert worker.name == "NewName"

    worker.start()
    worker.join()


def test_worker_daemon_property(backend):
    worker = backend.Worker(target=simple_task, daemon=False)
    assert worker.daemon is False

    worker.daemon = True
    assert worker.daemon is True

    worker.start()
    worker.join()


def test_worker_is_alive(backend):
    worker = backend.Worker(target=time.sleep, args=(0.01,))
    assert not worker.is_alive()

    worker.start()
    assert worker.is_alive()

    worker.join()
    assert not worker.is_alive()


def test_worker_join_timeout(backend):
    worker = backend.Worker(target=time.sleep, args=(0.1,))
    worker.start()

    # Join with short timeout should not wait for completion
    worker.join(timeout=0.01)
    assert worker.is_alive()

    # Join without timeout should wait
    worker.join()
    assert not worker.is_alive()


def test_worker_pickling_warning(backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        backend.Worker(target=lambda: None)
        assert len(w) == 1
        assert "not be picklable" in str(w[0].message)


def test_worker_local_function_warning(backend):
    def local_function():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        backend.Worker(target=local_function)
        assert len(w) == 1
        assert "not be picklable" in str(w[0].message)


def test_lock_acquire_release(backend):
    lock = backend.Lock()
    assert lock.acquire()
    lock.release()


def test_lock_context_manager(backend):
    lock = backend.Lock()
    assert not lock.locked()

    with lock:
        assert lock.locked()

    assert not lock.locked()


def test_lock_acquire_timeout(backend):
    lock = backend.Lock()

    acquired = lock.acquire(timeout=0.01)
    assert acquired

    acquired2 = lock.acquire(blocking=False)
    assert not acquired2

    lock.release()


def test_rlock_acquire_release(backend):
    lock = backend.RLock()
    assert lock.acquire()
    assert lock.acquire()
    lock.release()
    lock.release()


def test_rlock_context_manager(backend):
    lock = backend.RLock()
    with lock:
        with lock:
            assert lock is not None


def test_semaphore_acquire_release(backend):
    sem = backend.Semaphore(2)
    assert sem.acquire()
    sem.release()


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

    acquired = sem.acquire(timeout=0.01)
    assert acquired

    acquired2 = sem.acquire(blocking=False)
    assert not acquired2

    sem.release()


def test_bounded_semaphore_acquire_release(backend):
    sem = backend.BoundedSemaphore(2)
    assert sem.acquire()
    sem.release()


def test_bounded_semaphore_context_manager_exception(backend):
    sem = backend.BoundedSemaphore(1)
    try:
        with sem:
            raise ValueError("test")
    except ValueError:
        pass
    assert not sem.acquire(blocking=False) or sem.release() is None


def test_condition_acquire_release(backend):
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


def test_condition_wait(backend):
    cond = backend.Condition()
    error_raised = False
    try:
        with cond:
            cond.wait(timeout=0.01)
    except Exception:
        error_raised = True
    assert not error_raised


def test_condition_wait_for(backend):
    cond = backend.Condition()
    flag = []

    def predicate():
        return len(flag) > 0

    flag.append(1)

    with cond:
        result = cond.wait_for(predicate, timeout=0.01)
        assert result


def test_condition_notify_with_count(backend):
    cond = backend.Condition()
    assert cond.acquire()
    cond.notify(0)
    cond.notify(10)
    cond.release()
    assert cond.acquire()
    cond.release()


def test_event_is_set(backend):
    event = backend.Event()
    assert not event.is_set()

    event.set()
    assert event.is_set()

    event.clear()
    assert not event.is_set()


def test_event_wait(backend):
    event = backend.Event()

    # Wait on unset event should timeout
    result = event.wait(timeout=0.01)
    assert not result

    # Wait on set event should return immediately
    event.set()
    result = event.wait(timeout=0.01)
    assert result


def test_barrier_wait(backend):
    barrier = backend.Barrier(1)
    result = barrier.wait()
    assert result == 0


def test_barrier_wait_timeout(backend):
    barrier = backend.Barrier(2)
    timeout_raised = False

    try:
        barrier.wait(timeout=0.01)
    except threading.BrokenBarrierError:
        timeout_raised = True

    assert timeout_raised


def test_barrier_reset(backend):
    barrier = backend.Barrier(1)
    barrier.wait()
    barrier.reset()
    result = barrier.wait()
    assert result == 0


def test_barrier_abort(backend):
    barrier = backend.Barrier(2)
    barrier.abort()
    assert barrier.broken


def test_barrier_properties(backend):
    barrier = backend.Barrier(3)

    assert barrier.parties == 3
    assert barrier.n_waiting == 0
    assert not barrier.broken


def test_queue_put_get(backend):
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


def test_queue_task_done_join(backend):
    q = backend.Queue()
    error_raised = False

    try:
        q.put(1)
        q.put(2)

        q.get()
        q.task_done()
        q.get()
        q.task_done()

        q.join()
    except Exception:
        error_raised = True

    assert not error_raised


def test_queue_qsize(backend):
    q = backend.Queue()
    assert q.qsize() == 0
    q.put(1)

    if backend.get_backend() == "multiprocessing":
        time.sleep(0.01)

    assert q.qsize() == 1
    q.get()
    assert q.qsize() == 0


def test_queue_empty_full(backend):
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


def test_simple_queue_put_get(backend):
    q = backend.SimpleQueue()
    q.put(42)
    result = q.get()
    assert result == 42


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
            result = q.get(timeout=0.01)
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
            q.put(2, timeout=0.01)
            assert len(w) == 0

        result1 = q.get(block=False)
        result2 = q.get(timeout=0.01)
        assert result1 == 1
        assert result2 == 2


def test_pool_executor_map(backend):
    with backend.WorkerPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(square, [1, 2, 3, 4]))

    assert results == [1, 4, 9, 16]


def test_pool_executor_submit(backend):
    with backend.WorkerPoolExecutor(max_workers=2) as executor:
        future = executor.submit(square, 5)
        result = future.result()

    assert result == 25


def test_pool_executor_shutdown(backend):
    executor = backend.WorkerPoolExecutor(max_workers=2)

    # Submit a task
    future = executor.submit(square, 3)
    result = future.result()
    assert result == 9

    # Shutdown
    executor.shutdown(wait=True)

    # Cannot submit after shutdown
    try:
        executor.submit(square, 4)
        assert False, "Should not be able to submit after shutdown"
    except RuntimeError:
        pass
