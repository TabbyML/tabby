#pragma once

/**
 * @file BS_thread_pool_light.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (http://baraksh.com)
 * @version 3.5.0
 * @date 2023-05-25
 * @copyright Copyright (c) 2023 Barak Shoshany. Licensed under the MIT license. If you found this project useful, please consider starring it on GitHub! If you use this library in software of any kind, please provide a link to the GitHub repository https://github.com/bshoshany/thread-pool in the source code and documentation. If you use this library in published research, please cite it as follows: Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.5281/zenodo.4742687, arXiv:2105.00613 (May 2021)
 *
 * @brief BS::thread_pool_light: a fast, lightweight, and easy-to-use C++17 thread pool library. This header file contains a light version of the main library, for use when advanced features are not needed.
 */

#define BS_THREAD_POOL_LIGHT_VERSION "v3.5.0 (2023-05-25)"

#include <condition_variable> // std::condition_variable
#include <exception>          // std::current_exception
#include <functional>         // std::bind, std::function, std::invoke
#include <future>             // std::future, std::promise
#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
#include <queue>              // std::queue
#include <thread>             // std::thread
#include <type_traits>        // std::common_type_t, std::decay_t, std::invoke_result_t, std::is_void_v
#include <utility>            // std::forward, std::move, std::swap

namespace BS
{
/**
 * @brief A convenient shorthand for the type of std::thread::hardware_concurrency(). Should evaluate to unsigned int.
 */
using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;

/**
 * @brief A fast, lightweight, and easy-to-use C++17 thread pool class. This is a lighter version of the main thread pool class.
 */
class [[nodiscard]] thread_pool_light
{
public:
    // ============================
    // Constructors and destructors
    // ============================

    /**
     * @brief Construct a new thread pool.
     *
     * @param thread_count_ The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads.
     */
    thread_pool_light(const concurrency_t thread_count_ = 0) : thread_count(determine_thread_count(thread_count_)), threads(std::make_unique<std::thread[]>(determine_thread_count(thread_count_)))
    {
        create_threads();
    }

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads.
     */
    ~thread_pool_light()
    {
        wait_for_tasks();
        destroy_threads();
    }

    // =======================
    // Public member functions
    // =======================

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    [[nodiscard]] concurrency_t get_thread_count() const
    {
        return thread_count;
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. The user must use wait_for_tasks() or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T1 The type of the first index in the loop. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index in the loop. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.
     * @tparam T The common type of T1 and T2.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from first_index to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = first_index; i < index_after_last; ++i)". Note that if index_after_last == first_index, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     */
    template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
    void push_loop(T1 first_index_, T2 index_after_last_, F&& loop, size_t num_blocks = 0)
    {
        T first_index = static_cast<T>(first_index_);
        T index_after_last = static_cast<T>(index_after_last_);
        if (num_blocks == 0)
            num_blocks = thread_count;
        if (index_after_last < first_index)
            std::swap(index_after_last, first_index);
        size_t total_size = static_cast<size_t>(index_after_last - first_index);
        size_t block_size = static_cast<size_t>(total_size / num_blocks);
        if (block_size == 0)
        {
            block_size = 1;
            num_blocks = (total_size > 1) ? total_size : 1;
        }
        if (total_size > 0)
        {
            for (size_t i = 0; i < num_blocks; ++i)
                push_task(std::forward<F>(loop), static_cast<T>(i * block_size) + first_index, (i == num_blocks - 1) ? index_after_last : (static_cast<T>((i + 1) * block_size) + first_index));
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue. The user must use wait_for_tasks() or some other method to ensure that the loop finishes executing, otherwise bad things will happen. This overload is used for the special case where the first index is 0.
     *
     * @tparam F The type of the function to loop through.
     * @tparam T The type of the loop indices. Should be a signed or unsigned integer.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from 0 to (index_after_last - 1) inclusive. In other words, it will be equivalent to "for (T i = 0; i < index_after_last; ++i)". Note that if index_after_last == 0, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. loop(start, end) should typically involve a loop of the form "for (T i = start; i < end; ++i)".
     * @param num_blocks The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.
     */
    template <typename F, typename T>
    void push_loop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        push_loop(0, index_after_last, std::forward<F>(loop), num_blocks);
    }

    /**
     * @brief Push a function with zero or more arguments, but no return value, into the task queue. Does not return a future, so the user must use wait_for_tasks() or some other method to ensure that the task finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the arguments.
     * @param task The function to push.
     * @param args The zero or more arguments to pass to the function. Note that if the task is a class member function, the first argument must be a pointer to the object, i.e. &object (or this), followed by the actual arguments.
     */
    template <typename F, typename... A>
    void push_task(F&& task, A&&... args)
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks.push(std::bind(std::forward<F>(task), std::forward<A>(args)...)); // cppcheck-suppress ignoredReturnValue
        }
        task_available_cv.notify_one();
    }

    /**
     * @brief Submit a function with zero or more arguments into the task queue. If the function has a return value, get a future for the eventual returned value. If the function has no return value, get an std::future<void> which can be used to wait until the task finishes.
     *
     * @tparam F The type of the function.
     * @tparam A The types of the zero or more arguments to pass to the function.
     * @tparam R The return type of the function (can be void).
     * @param task The function to submit.
     * @param args The zero or more arguments to pass to the function. Note that if the task is a class member function, the first argument must be a pointer to the object, i.e. &object (or this), followed by the actual arguments.
     * @return A future to be used later to wait for the function to finish executing and/or obtain its returned value if it has one.
     */
    template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
    [[nodiscard]] std::future<R> submit(F&& task, A&&... args)
    {
        std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
        push_task(
            [task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...), task_promise]
            {
                try
                {
                    if constexpr (std::is_void_v<R>)
                    {
                        std::invoke(task_function);
                        task_promise->set_value();
                    }
                    else
                    {
                        task_promise->set_value(std::invoke(task_function));
                    }
                }
                catch (...)
                {
                    try
                    {
                        task_promise->set_exception(std::current_exception());
                    }
                    catch (...)
                    {
                    }
                }
            });
        return task_promise->get_future();
    }

    /**
     * @brief Wait for tasks to be completed, both those that are currently running in the threads and those that are still waiting in the queue. Note: To wait for just one specific task, use submit() instead, and call the wait() member function of the generated future.
     */
    void wait_for_tasks()
    {
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        tasks_done_cv.wait(tasks_lock, [this] { return !tasks_running && tasks.empty(); });
        waiting = false;
    }

private:
    // ========================
    // Private member functions
    // ========================

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     */
    void create_threads()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            workers_running = true;
        }
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread(&thread_pool_light::worker, this);
        }
    }

    /**
     * @brief Destroy the threads in the pool.
     */
    void destroy_threads()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            workers_running = false;
        }
        task_available_cv.notify_all();
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i].join();
        }
    }

    /**
     * @brief Determine how many threads the pool should have, based on the parameter passed to the constructor.
     *
     * @param thread_count_ The parameter passed to the constructor. If the parameter is a positive number, then the pool will be created with this number of threads. If the parameter is non-positive, or a parameter was not supplied (in which case it will have the default value of 0), then the pool will be created with the total number of hardware threads available, as obtained from std::thread::hardware_concurrency(). If the latter returns a non-positive number for some reason, then the pool will be created with just one thread.
     * @return The number of threads to use for constructing the pool.
     */
    [[nodiscard]] concurrency_t determine_thread_count(const concurrency_t thread_count_) const
    {
        if (thread_count_ > 0)
            return thread_count_;
        else
        {
            if (std::thread::hardware_concurrency() > 0)
                return std::thread::hardware_concurrency();
            else
                return 1;
        }
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool. Waits until it is notified by push_task() that a task is available, and then retrieves the task from the queue and executes it. Once the task finishes, the worker notifies wait_for_tasks() in case it is waiting.
     */
    void worker()
    {
        std::function<void()> task;
        while (true)
        {
            std::unique_lock tasks_lock(tasks_mutex);
            task_available_cv.wait(tasks_lock, [this] { return !tasks.empty() || !workers_running; });
            if (!workers_running)
                break;
            task = std::move(tasks.front());
            tasks.pop();
            ++tasks_running;
            tasks_lock.unlock();
            task();
            tasks_lock.lock();
            --tasks_running;
            if (waiting && !tasks_running && tasks.empty())
                tasks_done_cv.notify_all();
        }
    }

    // ============
    // Private data
    // ============

    /**
     * @brief A condition variable to notify worker() that a new task has become available.
     */
    std::condition_variable task_available_cv = {};

    /**
     * @brief A condition variable to notify wait_for_tasks() that the tasks are done.
     */
    std::condition_variable tasks_done_cv = {};

    /**
     * @brief A queue of tasks to be executed by the threads.
     */
    std::queue<std::function<void()>> tasks = {};

    /**
     * @brief A counter for the total number of currently running tasks.
     */
    size_t tasks_running = 0;

    /**
     * @brief A mutex to synchronize access to the task queue by different threads.
     */
    mutable std::mutex tasks_mutex = {};

    /**
     * @brief The number of threads in the pool.
     */
    concurrency_t thread_count = 0;

    /**
     * @brief A smart pointer to manage the memory allocated for the threads.
     */
    std::unique_ptr<std::thread[]> threads = nullptr;

    /**
     * @brief A flag indicating that wait_for_tasks() is active and expects to be notified whenever a task is done.
     */
    bool waiting = false;

    /**
     * @brief A flag indicating to the workers to keep running. When set to false, the workers terminate permanently.
     */
    bool workers_running = false;
};

} // namespace BS
