# `freethreading` — Thread-first true parallelism

[![CI Build](https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=LWBRgtlX8j)](https://codecov.io/gh/iskandergaba/freethreading)
[![Docs](https://readthedocs.org/projects/freethreading/badge/?version=latest)](https://freethreading.readthedocs.io/en/latest)

---

`freethreading` is a lightweight wrapper that provides a unified API for true parallel execution in Python. It
automatically uses `threading` on free-threaded Python builds (where the [Global Interpreter Lock (GIL)](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
is disabled) and falls back to `multiprocessing` on standard ones. This enables true parallelism across Python versions
while preferring the efficiency of [threads](https://en.wikipedia.org/wiki/Thread_(computing)) over
[processes](https://en.wikipedia.org/wiki/Process_(computing)) whenever possible.

## Installation

<!-- To install `freethreading`, simply run:

```shell
pip install freethreading
``` -->

To install the latest development version, you can run:

```shell
pip install git+https://github.com/iskandergaba/freethreading.git
```

## Quick Start

`freethreading` is a drop-in replacement for *most* pre-existing `threading` and `multiprocessing` code. To achieve
this, the module exposes only non-deprecated common functionality shared between both backends while discarding any
backend-specific APIs. The following examples show how to get started.

`freethreading` remains consistent with the standard library, so wrapper classes work as drop-in replacements for those
used by `threading` and `multiprocessing`. Here's how they work:

```python
# threading
from queue import Queue
from threading import Event, Lock

# multiprocessing
from multiprocessing import Event, Lock, Queue

# freethreading (replaces both)
from freethreading import Event, Lock, Queue
event = Event()
lock = Lock()
queue = Queue()
lock.acquire()
# True
event.set()
queue.put("data")
event.is_set()
# True
queue.get()
# 'data'
lock.release()
```

`freethreading` functions merge as much functionality from both backends as possible to ensure consistent behavior
across backends and simplify adoption. Here's what that looks like:

```python
# threading
from threading import enumerate, get_ident

# multiprocessing
from multiprocessing import active_children
from os import getpid

# freethreading (replaces both)
from freethreading import active_children, enumerate, get_ident
len(active_children())  # excludes current thread or process
# 0
len(enumerate())  # includes current thread or process
# 1
get_ident()  # current thread or process identifier
# 140247834...
```

Only `Worker`, `WorkerPool`, `WorkerPoolExecutor`, and `current_worker` differ from the standard library naming, using
"worker" as a term for both threads and processes. Here is an example:

```python
# threading
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, current_thread

# multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, current_process

# freethreading (replaces both)
from freethreading import Worker, WorkerPool, WorkerPoolExecutor, current_worker
current_worker().name  # 'MainThread' or 'MainProcess'
# 'MainThread'

def task():
    print(f"Hello from {current_worker().name}!")

# Using Worker (Thread or Process) to run a task
w = Worker(target=task, name="MyWorker")
w.start()
w.join()
# Hello from MyWorker!

# Using WorkerPool (Pool or ThreadPool) to distribute work
def square(x):
    return x * x

with WorkerPool(workers=2) as pool:
    print(pool.map(square, range(5)))
# [0, 1, 4, 9, 16]

# Using WorkerPoolExecutor (ThreadPoolExecutor or ProcessPoolExecutor) to run a task
with WorkerPoolExecutor(max_workers=2) as executor:
    # 'Hello from ThreadPoolExecutor-0_0!' or 'Hello from ForkProcess-2!'
    future = executor.submit(task)
# Hello from ThreadPoolExecutor-0_0!
```

## Documentation

For more details, check out the full documentation at [freethreading.readthedocs.io](https://freethreading.readthedocs.io).
