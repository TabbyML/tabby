#pragma once

#include <random>

namespace ctranslate2 {

  void set_random_seed(const unsigned int seed);
  unsigned int get_random_seed();
  std::mt19937& get_random_generator();

}
