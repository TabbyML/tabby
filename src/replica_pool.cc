#include "ctranslate2/replica_pool.h"

#include "env.h"

namespace ctranslate2 {

  ReplicaWorker::ReplicaWorker(Device device, int device_index, size_t num_threads)
    : _device(device)
    , _device_index(device_index)
    , _num_threads(num_threads)
    , _allocator(nullptr)
  {
  }

  void ReplicaWorker::initialize() {
    // Set the number of OpenMP threads for the current thread.
    set_num_threads(_num_threads);

    // Register the memory allocator used in this thread.
    _allocator = &get_allocator(_device);
  }

  void ReplicaWorker::idle() {
    // When no new jobs are immediately available, we synchronize the CUDA device
    // so that the CudaAsyncAllocator can release some memory.
    synchronize_device(_device, _device_index);
  }


  ReplicaPool::ReplicaPool(std::vector<std::unique_ptr<Worker>> workers,
                           const long max_queued_batches)
  {
    size_t max_queue_size = std::numeric_limits<size_t>::max();
    if (max_queued_batches == 0)
      max_queue_size = 4 * workers.size();
    else if (max_queued_batches > 0)
      max_queue_size = max_queued_batches;

    static const int core_offset = read_int_from_env("CT2_TRANSLATORS_CORE_OFFSET", -1);

    _thread_pool = std::make_unique<ThreadPool>(std::move(workers), max_queue_size, core_offset);
  }

  size_t ReplicaPool::num_queued_batches() const {
    return _thread_pool->num_queued_jobs();
  }

  size_t ReplicaPool::num_active_batches() const {
    return _thread_pool->num_active_jobs();
  }

  size_t ReplicaPool::num_replicas() const {
    return _thread_pool->num_threads();
  }

}
