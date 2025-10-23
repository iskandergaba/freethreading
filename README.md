<div align="center">
<h1>Freethreading</h1>

[![Codecov](https://codecov.io/gh/iskandergaba/freethreading/graph/badge.svg?token=LWBRgtlX8j)](https://codecov.io/gh/iskandergaba/freethreading)
[![Docs](https://readthedocs.org/projects/freethreading/badge/?version=latest)](https://freethreading.readthedocs.io/en/latest)
[![CI Build](https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandergaba/freethreading/actions/workflows/ci.yml)

**Automatic selection between threading and multiprocessing based on Python's GIL status.**

Freethreading provides a unified concurrency API that automatically uses `threading` when the GIL is disabled (free-threaded Python 3.13+) and `multiprocessing` when the GIL is enabled (standard Python). Write once, run optimally everywhere.
</div>

## Installation

To install `freethreading`, simply run:

```shell
pip install freethreading
```

To install the latest development version:

```shell
pip install git+https://github.com/iskandergaba/freethreading.git
```

## Quick Start

```python
import freethreading

# Check which backend is being used
print(freethreading.get_backend())  # 'threading' or 'multiprocessing'

# Use unified API - automatically selects best backend
from freethreading import Worker, Lock, Queue

# Create synchronization primitives in main scope
q = Queue()
lock = Lock()

def worker_function():
    """Worker accesses queue and lock from outer scope."""
    with lock:
        q.put("Hello from worker!")

worker = Worker(target=worker_function)
worker.start()
worker.join()

print(q.get())  # 'Hello from worker!'
```

## Documentation

Full documentation is available at [freethreading.readthedocs.io](https://freethreading.readthedocs.io).

## Features

- **Automatic backend selection** based on GIL status
- **Unified API** across threading and multiprocessing
- **Zero-overhead** abstraction with runtime backend selection
- **Complete synchronization primitives**: Locks, Semaphores, Events, Barriers, Conditions
- **Thread-safe queues** with full compatibility
- **Worker management** with unified Thread/Process interface
- **Executor support** via PoolExecutor

## When to Use

✅ **Good for:**
- Projects targeting both standard and free-threaded Python
- CPU-bound parallel workloads across Python versions
- Libraries providing automatic performance optimization

❌ **Not for:**
- Projects targeting only one Python variant (use `threading` or `multiprocessing` directly)
- Applications requiring features beyond the common subset
- Real-time systems needing predictable behavior

See the [Caveats & Limitations](https://freethreading.readthedocs.io/en/latest/caveats.html) documentation for more details.

## License

MIT License - see [LICENSE](LICENSE) file for details.
