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

void benchmark_split(Device device) {
  StorageView x({64, 512*3}, DataType::DT_FLOAT, device);
  StorageView a(device);
  StorageView b(device);
  StorageView c(device);
  const ops::Split split_op(-1);
  BENCHMARK(split_op(x, a, b, c), 10000);
}

void benchmark_layer_norm(Device device) {
  std::vector<float> gamma_ = rand_vector(512);
  std::vector<float> beta_ = rand_vector(512);
  std::vector<float> x_ = rand_vector(100 * 512);

  StorageView gamma({512}, gamma_, device);
  StorageView beta({512}, beta_, device);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::LayerNorm layer_norm_op;
  BENCHMARK(layer_norm_op(beta, gamma, x, y), 10000);
}

void benchmark_softmax(Device device) {
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op;
  BENCHMARK(softmax_op(x, y), 10000);
}

void benchmark_topk(Device device) {
  const size_t k = 4;
  const size_t batch_size = 8;
  const size_t vocab_size = 32000;
  std::vector<float> x = rand_vector(batch_size * k * vocab_size);
  StorageView input({batch_size, k * vocab_size}, x, device);
  StorageView values(input.dtype(), device);
  StorageView indices(DataType::DT_INT32,  device);
  const ops::TopK op(k);
  BENCHMARK(op(input, values, indices), 10000);
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
  else if (op == "transpose")
    benchmark_transpose(device);
  else if (op == "split")
    benchmark_split(device);
  else if (op == "layer_norm")
    benchmark_layer_norm(device);
  else if (op == "softmax")
    benchmark_softmax(device);
  else if (op == "topk")
    benchmark_topk(device);

  return 0;
}
