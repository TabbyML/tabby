#include "benchmark_utils.h"

#include <numeric>

#include "ctranslate2/ops/ops.h"

using namespace ctranslate2;

void benchmark_gather(Device device) {
  StorageView data({512, 512}, DataType::FLOAT32, device);
  std::vector<int32_t> input_v(250);
  std::iota(input_v.begin(), input_v.end(), 0);
  StorageView input({static_cast<dim_t>(input_v.size())}, input_v, device);
  StorageView output(device);
  const ops::Gather gather_op;
  BENCHMARK(gather_op(data, input, output), 100000);
}

void benchmark_transpose(Device device) {
  StorageView x({64, 48, 8, 64}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Transpose transpose_op({0, 2, 1, 3});
  BENCHMARK(transpose_op(x, y), 1000);
}

void benchmark_split(Device device) {
  StorageView x({64, 512*3}, DataType::FLOAT32, device);
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

void benchmark_masked_softmax(Device device) {
  const dim_t batch_size = 32;
  const dim_t num_heads = 8;
  const dim_t max_source = 24;
  const dim_t max_target = 36;
  StorageView lengths({batch_size}, std::vector<int32_t>(batch_size, max_source - 5), device);
  StorageView x({batch_size, num_heads, max_source, max_target},
                rand_vector(batch_size * num_heads * max_source * max_target),
                device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, lengths, y), 10000);
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
  DataType output_dtype = dtype != DataType::FLOAT32 ? DataType::INT32 : dtype;
  StorageView a({32 * 32, 512}, dtype, device);
  StorageView b({2048, 512}, dtype, device);
  StorageView c(output_dtype, device);
  const ops::Gemm gemm_op(1, 0, false, true);
  BENCHMARK(gemm_op(a, b, c), 1000);
}

void benchmark_quantize(Device device, DataType dtype) {
  StorageView x({32, 512}, rand_vector(32 * 512), device);
  StorageView y(dtype, device);
  StorageView scale(DataType::FLOAT32, device);
  const ops::Quantize quantize_op;
  BENCHMARK(quantize_op(x, y, scale), 10000);
}

void benchmark_dequantize(Device device) {
  StorageView x({64, 8192}, DataType::INT32, device);
  StorageView input_scale({32}, DataType::FLOAT32, device);
  StorageView weight_scale({8192}, DataType::FLOAT32, device);
  StorageView bias({8192}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::ActivationType activation_type = ops::ActivationType::ReLU;
  const ops::Dequantize dequantize_op(&activation_type);
  BENCHMARK(dequantize_op(x, input_scale, weight_scale, false, true, y, &bias), 10000);
}

void benchmark_conv1d(Device device) {
  StorageView x({1, 768, 3000}, DataType::FLOAT32, device);
  StorageView weight({768, 768, 3}, DataType::FLOAT32, device);
  StorageView bias({768}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Conv1D conv_op{2, 1};
  BENCHMARK(conv_op(x, weight, bias, y), 100);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device [dtype]" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  Device device = std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU;
  std::string dtype_str = argc > 3 ? argv[3] : "float32";
  DataType dtype = DataType::FLOAT32;
  if (dtype_str == "int16")
    dtype = DataType::INT16;
  else if (dtype_str == "int8")
    dtype = DataType::INT8;

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
  else if (op == "masked_softmax")
    benchmark_masked_softmax(device);
  else if (op == "topk")
    benchmark_topk(device);
  else if (op == "gemm")
    benchmark_gemm(device, dtype);
  else if (op == "quantize")
    benchmark_quantize(device, dtype);
  else if (op == "dequantize")
    benchmark_dequantize(device);
  else if (op == "conv1d")
    benchmark_conv1d(device);

  return 0;
}
