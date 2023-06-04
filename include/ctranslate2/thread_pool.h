#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ctranslate2 {

  // Base class for asynchronous jobs.
  class Job {
  public:
    virtual ~Job();
    virtual void run() = 0;

    // The job counter is used to track the number of active jobs (queued and currently processed).
    void set_job_counter(std::atomic<size_t>& counter);

  private:
    std::atomic<size_t>* _counter = nullptr;
  };

  // A thread-safe queue of jobs.
  class JobQueue {
  public:
    JobQueue(size_t maximum_size);
    ~JobQueue();

    size_t size() const;

    // Puts a job in the queue. The method blocks until a free slot is available.
    void put(std::unique_ptr<Job> job);

    // Gets a job from the queue. The method blocks until a job is available.
    // If the queue is closed, the method returns a null pointer.
    std::unique_ptr<Job> get(const std::function<void()>& before_wait = nullptr);

    void close();

  private:
    bool can_get_job() const;

    mutable std::mutex _mutex;
    std::queue<std::unique_ptr<Job>> _queue;
    std::condition_variable _can_put_job;
    std::condition_variable _can_get_job;
    size_t _maximum_size;
    bool _request_end;
  };

  // A worker processing jobs in a thread.
  class Worker {
  public:
    virtual ~Worker() = default;

    void start(JobQueue& job_queue, int thread_affinity = -1);
    void join();

  protected:
    // Called before the work loop.
    virtual void initialize() {}

    // Called after the work loop.
    virtual void finalize() {}

    // Called before waiting for new jobs.
    virtual void idle() {}

  private:
    void run(JobQueue& job_queue);

    std::thread _thread;
  };

  // A pool of threads.
  class ThreadPool {
  public:
    // Default thread workers.
    ThreadPool(size_t num_threads,
               size_t maximum_queue_size = std::numeric_limits<size_t>::max(),
               int core_offset = -1);

    // User-defined thread workers.
    ThreadPool(std::vector<std::unique_ptr<Worker>> workers,
               size_t maximum_queue_size = std::numeric_limits<size_t>::max(),
               int core_offset = -1);

    ~ThreadPool();

    // Posts a new job. The method blocks if the job queue is full.
    void post(std::unique_ptr<Job> job);

    size_t num_threads() const;

    // Number of jobs in the queue.
    size_t num_queued_jobs() const;

    // Number of jobs in the queue and currently processed by a worker.
    size_t num_active_jobs() const;

    Worker& get_worker(size_t index);
    static Worker& get_local_worker();

  private:
    void start_workers(int core_offset);

    JobQueue _queue;
    std::vector<std::unique_ptr<Worker>> _workers;
    std::atomic<size_t> _num_active_jobs;
  };

}
