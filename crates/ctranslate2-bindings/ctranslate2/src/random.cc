#include "ctranslate2/random.h"

#include <atomic>

namespace ctranslate2 {

  constexpr unsigned int default_seed = static_cast<unsigned int>(-1);
  static std::atomic<unsigned int> g_seed(default_seed);

  void set_random_seed(const unsigned int seed) {
    g_seed = seed;
  }

  unsigned int get_random_seed() {
    return g_seed == default_seed ? std::random_device{}() : g_seed.load();
  }

  std::mt19937& get_random_generator() {
    static thread_local std::mt19937 generator(get_random_seed());
    return generator;
  }

}
