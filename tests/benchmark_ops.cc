#include "benchmark_utils.h"

#include "ctranslate2/ops/ops.h"

using namespace ctranslate2;

void benchmark_gather(Device device) {
  StorageView data({512, 512}, DataType::DT_FLOAT, device);
  std::vector<int32_t> input_v(250);
  std::iota(input_v.begin(), input_v.end(), 0);
  StorageView input({input_v.size()}, input_v);
  StorageView output(device);
  const ops::Gather gather_op;
  BENCHMARK(gather_op(data, input, output), 100000);
}

void benchmark_transpose(Device device) {
  StorageView x({64, 48, 512}, DataType::DT_FLOAT, device);
  StorageView y(device);
  const ops::Transpose transpose_op({2, 1, 0});
  BENCHMARK(transpose_op(x, y), 1000);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  Device device = std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU;

  if (op == "gather")
    benchmark_gather(device);
  if (op == "transpose")
    benchmark_transpose(device);

  return 0;
}
