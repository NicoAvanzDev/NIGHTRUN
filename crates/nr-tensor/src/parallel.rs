//! Spin-worker pool for splitting matvec rows across cores.
//!
//! Workers (UEFI application processors on bare metal, std threads on the
//! host) call `POOL.worker_loop()` once and never return; the main core
//! posts jobs and participates in the work. Workers use only atomics and
//! compute — no firmware/OS calls — so the same code runs everywhere.

use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};

struct Task<'a> {
    run: &'a (dyn Fn(usize, usize) + Sync),
}

pub struct Pool {
    seq: AtomicUsize,
    cursor: AtomicUsize,
    total: AtomicUsize,
    chunk: AtomicUsize,
    done: AtomicUsize,
    task: AtomicUsize,
    workers: AtomicUsize,
    ready: AtomicUsize,
}

pub static POOL: Pool = Pool::new();

impl Pool {
    const fn new() -> Pool {
        Pool {
            seq: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            chunk: AtomicUsize::new(1),
            done: AtomicUsize::new(0),
            task: AtomicUsize::new(0),
            workers: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
        }
    }

    /// How many workers have entered their loop.
    pub fn ready_workers(&self) -> usize {
        self.ready.load(SeqCst)
    }

    /// Activate the pool once `n` workers are spinning.
    pub fn activate(&self, n: usize) {
        assert!(self.ready.load(SeqCst) >= n, "workers not ready");
        self.workers.store(n, SeqCst);
    }

    pub fn active_workers(&self) -> usize {
        self.workers.load(SeqCst)
    }

    /// Entry point for worker threads/APs. Never returns.
    pub fn worker_loop(&self) -> ! {
        self.ready.fetch_add(1, SeqCst);
        let mut last = self.seq.load(SeqCst);
        loop {
            let s = self.seq.load(SeqCst);
            if s != last {
                last = s;
                // SAFETY: `run` keeps the Task alive until every worker
                // has bumped `done` for this seq.
                let task = unsafe { &*(self.task.load(SeqCst) as *const Task) };
                self.drain(task);
                self.done.fetch_add(1, SeqCst);
            }
            core::hint::spin_loop();
        }
    }

    fn drain(&self, task: &Task) {
        let total = self.total.load(SeqCst);
        let chunk = self.chunk.load(SeqCst).max(1);
        loop {
            let start = self.cursor.fetch_add(chunk, SeqCst);
            if start >= total {
                break;
            }
            (task.run)(start, (start + chunk).min(total));
        }
    }

    /// Split `0..total` across the pool; `run(start, end)` must be safe to
    /// call concurrently on disjoint ranges. Runs inline when no workers.
    pub fn run(&self, total: usize, run: &(dyn Fn(usize, usize) + Sync)) {
        let n = self.workers.load(SeqCst);
        if n == 0 || total < 128 {
            run(0, total);
            return;
        }
        let task = Task { run };
        self.total.store(total, SeqCst);
        self.cursor.store(0, SeqCst);
        self.chunk.store((total / ((n + 1) * 4)).max(16), SeqCst);
        self.done.store(0, SeqCst);
        self.task.store(&task as *const Task as usize, SeqCst);
        self.seq.fetch_add(1, SeqCst);
        self.drain(&task);
        while self.done.load(SeqCst) < n {
            core::hint::spin_loop();
        }
    }
}
