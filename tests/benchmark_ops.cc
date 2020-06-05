#include "benchmark_utils.h"

#include <numeric>

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/utils.h"

using namespace ctranslate2;

void benchmark_gather(Device device) {
  StorageView data({512, 512}, DataType::FLOAT, device);
  std::vector<int32_t> input_v(250);
  std::iota(input_v.begin(), input_v.end(), 0);
  StorageView input({static_cast<dim_t>(input_v.size())}, input_v, device);
  StorageView output(device);
  const ops::Gather gather_op;
  BENCHMARK(gather_op(data, input, output), 100000);
}

void benchmark_transpose(Device device) {
  StorageView x({64, 48, 8, 64}, DataType::FLOAT, device);
  StorageView y(device);
  const ops::Transpose transpose_op({0, 2, 1, 3});
  BENCHMARK(transpose_op(x, y), 1000);
}

void benchmark_split(Device device) {
  StorageView x({64, 512*3}, DataType::FLOAT, device);
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
  const ops::LayerNorm layer_norm_op{};
  BENCHMARK(layer_norm_op(beta, gamma, x, y), 10000);
}

void benchmark_softmax(Device device) {
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, y), 10000);
}

void benchmark_topk(Device device) {
  const size_t k = 4;
  const size_t batch_size = 8;
  const size_t vocab_size = 32000;
  std::vector<float> x = rand_vector(batch_size * k * vocab_size);
  StorageView input({batch_size, k * vocab_size}, x, device);
  StorageView values(input.dtype(), device);
  StorageView indices(DataType::INT32,  device);
  const ops::TopK op(k);
  BENCHMARK(op(input, values, indices), 2000);
}

void benchmark_gemm(Device device, DataType dtype) {
  DataType output_dtype = dtype != DataType::FLOAT ? DataType::INT32 : dtype;
  StorageView a({32 * 32, 512}, dtype, device);
  StorageView b({2048, 512}, dtype, device);
  StorageView c(output_dtype, device);
  const ops::Gemm gemm_op(1, 0, false, true);
  BENCHMARK(gemm_op(a, b, c), 1000);
}

void benchmark_quantize(Device device, DataType dtype) {
  StorageView x({32, 512}, rand_vector(32 * 512), device);
  StorageView y(dtype, device);
  StorageView scale(DataType::FLOAT, device);
  const ops::Quantize quantize_op;
  BENCHMARK(quantize_op(x, y, scale), 10000);
}

void benchmark_dequantize(Device device) {
  StorageView x({32, 1536}, DataType::INT32, device);
  StorageView input_scale({32}, DataType::FLOAT, device);
  StorageView weight_scale({1536}, DataType::FLOAT, device);
  StorageView y(device);
  const ops::Dequantize dequantize_op{};
  BENCHMARK(dequantize_op(x, input_scale, weight_scale, false, true, y), 100000);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device [dtype]" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  Device device = std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU;
  std::string dtype_str = argc > 3 ? argv[3] : "float";
  DataType dtype = DataType::FLOAT;
  if (dtype_str == "int16")
    dtype = DataType::INT16;
  else if (dtype_str == "int8")
    dtype = DataType::INT8;

  ctranslate2::set_num_threads(4);

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
  else if (op == "gemm")
    benchmark_gemm(device, dtype);
  else if (op == "quantize")
    benchmark_quantize(device, dtype);
  else if (op == "dequantize")
    benchmark_dequantize(device);

  return 0;
}
