#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include <mkl.h>

#include <mkldnn.hpp>

class Variable;
using VariableIndex = std::map<std::string, Variable>;
using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

class Vocabulary
{
public:
  static const std::string unk_token;

  Vocabulary(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      _token_to_id.emplace(line, _id_to_token.size());
      _id_to_token.push_back(line);
    }
    _token_to_id.emplace(unk_token, _id_to_token.size());
    _id_to_token.push_back(unk_token);

  }

  const std::string& to_token(unsigned int id) const {
    return _id_to_token[id];
  }
  unsigned int to_id(const std::string& token) const {
    auto it = _token_to_id.find(token);
    if (it == _token_to_id.end())
      return _token_to_id.at(unk_token);
    return it->second;
  }
  unsigned int size() const {
    return _id_to_token.size();
  }

private:
  std::vector<std::string> _id_to_token;
  std::unordered_map<std::string, unsigned int> _token_to_id;
};

const std::string Vocabulary::unk_token = "<unk>";

class Variable
{
public:
  Variable(unsigned int rank,
              const unsigned int* dimensions,
              const float* data)
    : _rank(rank)
    , _dimensions(dimensions)
    , _data(data) {
  }
  unsigned int rank() const {
    return _rank;
  }
  const unsigned int* dim() const {
    return _dimensions;
  }
  const float* data() const {
    return _data;
  }

  friend std::ostream& operator<<(std::ostream& os, const Variable& index);

private:
  unsigned int _rank;
  const unsigned int* _dimensions;
  const float* _data;
};

std::ostream& operator<<(std::ostream& os, const Variable& index) {
  os << '(';
  for (unsigned int i = 0; i < index._rank; ++i) {
    if (i > 0)
      os << ", ";
    os << index._dimensions[i];
  }
  os << ')';
  return os;
}

const Variable& get_variable(const VariableIndex& index, const std::string& scope) {
  auto it = index.lower_bound(scope);
  return it->second;
}
const float* get_variable_data(const VariableIndex& index, const std::string& scope) {
  return get_variable(index, scope).data();
}

template <typename T>
T consume(unsigned char** ptr) {
  T val = *reinterpret_cast<T*>(*ptr);
  *ptr += sizeof(T);
  return val;
}

VariableIndex build_variable_index(void* model) {
  auto ptr = reinterpret_cast<unsigned char*>(model);
  auto num_variables = consume<unsigned int>(&ptr);

  VariableIndex variable_index;

  for (unsigned int i = 0; i < num_variables; ++i) {
    auto name_length = consume<unsigned short>(&ptr);
    auto name = reinterpret_cast<const char*>(ptr);
    ptr += name_length;
    unsigned int rank = consume<unsigned char>(&ptr);
    auto dimensions = reinterpret_cast<const unsigned int*>(ptr);
    unsigned int offset = 1;
    for (unsigned int k = 0; k < rank; k++)
      offset *= consume<unsigned int>(&ptr);
    unsigned int data_width = consume<unsigned char>(&ptr);
    auto data = reinterpret_cast<const float*>(ptr);
    ptr += offset * data_width;
    variable_index.emplace(name, Variable(rank, dimensions, data));
  }

  return variable_index;
}

void* load_model(const char* path) {
  struct stat st;
  int s = stat(path, &st);
  if (s == -1)
    return nullptr;
  int fd = open(path, O_RDONLY, 0);
  if (fd == -1)
    return nullptr;
  void* model = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  close(fd);
  if (model == MAP_FAILED)
    return nullptr;
  return model;
}

void maybe_resize(std::vector<float>& v, size_t size) {
  if (size > v.size()) {
    v.resize(size);
  }
}

float array_sum(const float* array, unsigned int size) {
  float sum = 0;
  for (unsigned int i = 0; i < size; ++i)
    sum += array[i];
  return sum;
}

float array_mean(const float* array, unsigned int size) {
  return array_sum(array, size) / size;
}

void sgemm(const float* a,
           const float* b,
           CBLAS_TRANSPOSE trans_a,
           CBLAS_TRANSPOSE trans_b,
           MKL_INT m,
           MKL_INT n,
           MKL_INT k,
           float beta,
           float* c) {
  MKL_INT lda = (trans_a == CblasNoTrans) ? k : m;
  MKL_INT ldb = (trans_b == CblasNoTrans) ? n : k;
  MKL_INT ldc = n;

  cblas_sgemm(CblasRowMajor, trans_a, trans_b,
              m, n, k,
              1.0 /* alpha */, a, lda,
              b, ldb,
              beta, c, ldc);
}

void mat_mul(const float* a,
             const float* b,
             CBLAS_TRANSPOSE trans_a,
             CBLAS_TRANSPOSE trans_b,
             MKL_INT m,
             MKL_INT n,
             MKL_INT k,
             float* c) {
  sgemm(a, b,
        trans_a, trans_b,
        m, n, k,
        0.0 /* beta */, c);
}

void batch_mat_mul(const float* a,
                   const float* b,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   MKL_INT batch_size,
                   MKL_INT m,
                   MKL_INT n,
                   MKL_INT k,
                   float* c) {
  MKL_INT lda = (trans_a == CblasNoTrans) ? k : m;
  MKL_INT ldb = (trans_b == CblasNoTrans) ? n : k;
  MKL_INT ldc = n;
  float alpha = 1.0;
  float beta = 0.0;

  std::vector<const float*> a_array(batch_size);
  std::vector<const float*> b_array(batch_size);
  std::vector<float*> c_array(batch_size);
  for (MKL_INT i = 0; i < batch_size; ++i) {
    a_array[i] = a + (i * m * k);
    b_array[i] = b + (i * k * n);
    c_array[i] = c + (i * m * n);
  }

  cblas_sgemm_batch(CblasRowMajor,
                    &trans_a, &trans_b,
                    &m, &n, &k,
                    &alpha, a_array.data(), &lda,
                    b_array.data(), &ldb,
                    &beta, c_array.data(), &ldc,
                    1 /* group_count */, &batch_size);
}

void concat_in_depth(const std::vector<const float*>& inputs,
                     const std::vector<unsigned int>& depths,
                     unsigned int batch_size,
                     float* output) {
  unsigned int num_inputs = inputs.size();
  unsigned int total_depth = 0;

  for (unsigned int i = 0; i < num_inputs; ++i) {
    const unsigned int depth = depths[i];
    const float* a = inputs[i];
    float* b = output + (total_depth * batch_size);
    mkl_somatcopy('R', 'T', batch_size, depth, 1.0, a, depth, b, batch_size);
    total_depth += depth;
  }

  mkl_simatcopy('R', 'T', total_depth, batch_size, 1.0, output, batch_size, total_depth);
}


class Dense
{
public:
  Dense(const std::map<std::string, Variable>& index, const std::string& scope)
    : _weight(get_variable(index, scope + "/kernel"))
    , _bias(get_variable(index, scope + "/bias")) {
  }

  void compute(const float* input,
               unsigned int batch_size,
               unsigned int input_depth,
               float* output) {
    MKL_INT m = batch_size;
    MKL_INT n = output_depth();
    MKL_INT k = input_depth;
    for (int i = 0; i < m; ++i)
      memcpy(output + (i * n), _bias.data(), n * sizeof (float));
    sgemm(input, _weight.data(),
          CblasNoTrans, CblasNoTrans,
          m, n, k,
          1.0, output);
  }

  unsigned int output_depth() const {
    return _weight.dim()[2];
  }

private:
  const Variable& _weight;
  const Variable& _bias;
};

class LayerNorm
{
public:
  LayerNorm(const std::map<std::string, Variable>& index, const std::string& scope)
    : _beta(get_variable_data(index, scope + "/beta"))
    , _gamma(get_variable_data(index, scope + "/gamma")) {
  }

  void compute(const float* input,
               unsigned int batch_size,
               unsigned int depth,
               float* output) {
    maybe_resize(_tmp, depth);
    for (unsigned int i = 0; i < batch_size; ++i) {
      const float* x = input + (i * depth);
      float* y = output + (i * depth);
      float mean = array_mean(x, depth);
      memcpy(y, x, depth * sizeof (float));
      cblas_saxpy(depth, -1.0, &mean, 0, y, 1); // y is now centered
      vsPowx(depth, y, 2, _tmp.data());
      float std = array_mean(_tmp.data(), depth);
      cblas_sscal(depth, std, y, 1); // y is now centered and normalized.
      vsMul(depth, y, _beta, y);
      cblas_saxpy(depth, 1.0, _gamma, 1, y, 1);
    }
  }

private:
  const float* _beta;
  const float* _gamma;
  std::vector<float> _tmp;
};

void softmax(mkldnn::engine& engine, float* input, int batch, int depth, float* output) {
  auto memory_desc = mkldnn::memory::desc({batch, depth},
                                          mkldnn::memory::data_type::f32,
                                          mkldnn::memory::format::nc);
  auto softmax_desc = mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_inference,
                                                    memory_desc, 1 /* softmax_axis */);

  auto memory_primitive_desc = mkldnn::memory::primitive_desc(memory_desc, engine);
  auto input_memory = mkldnn::memory(memory_primitive_desc, input);
  auto output_memory = mkldnn::memory(memory_primitive_desc, output);

  std::vector<mkldnn::primitive> primitives;
  primitives.push_back(mkldnn::softmax_forward(
                         mkldnn::softmax_forward::primitive_desc(softmax_desc, engine),
                         mkldnn::primitive::at(input_memory),
                         output_memory));
  mkldnn::stream(mkldnn::stream::kind::eager).submit(primitives).wait();
}

class DotProductAttention
{
public:
  DotProductAttention(mkldnn::engine& engine)
    : _engine(engine) {
  }

  void operator()(const float* queries,
                  const float* keys,
                  const float* values,
                  unsigned int batch,
                  unsigned int queries_time,
                  unsigned int keys_time,
                  unsigned int depth,
                  float* context) {
    maybe_resize(_dot, batch * queries_time * keys_time);
    maybe_resize(_attn, batch * queries_time * keys_time);

    batch_mat_mul(queries, keys,
                  CblasNoTrans, CblasTrans,
                  batch, queries_time, keys_time, depth,
                  _dot.data());
    softmax(_engine, _dot.data(), batch * queries_time, keys_time, _attn.data());
    batch_mat_mul(_attn.data(), values,
                  CblasNoTrans, CblasNoTrans,
                  batch, queries_time, depth, keys_time,
                  context);
  }

private:
  mkldnn::engine _engine;
  std::vector<float> _dot;
  std::vector<float> _attn;
};

// class SoftMax
// {
// public:
//   SoftMax(mkldnn::engine& engine)
//     : _engine(engine) {
//   }

//   template <typename Input, typename Output>
//   void compute(const Input& input, Output& output) {
//     const auto& dims = input.dimensions();

//     int D = dims.back();
//   }

//   virtual std::vector<float>& apply(const float* input,
//                                     const std::vector<unsigned int>& dims) override {
//     int D = dims.back();
//     int B = 1;
//     for (size_t i = 0; i < dims.size() - 1; ++i)
//       B *= dims[i];

//     _output.resize(B * D);

//     auto memory_desc = mkldnn::memory::desc({B, D},
//                                             mkldnn::memory::data_type::f32,
//                                             mkldnn::memory::format::nc);
//     auto softmax_desc = mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_inference,
//                                                       memory_desc, 1 /* softmax_axis */);

//     auto memory_primitive_desc = mkldnn::memory::primitive_desc(memory_desc, _engine);
//     auto input_memory = mkldnn::memory(memory_primitive_desc, const_cast<float*>(input));
//     auto output_memory = mkldnn::memory(memory_primitive_desc, _output.data());

//     std::vector<mkldnn::primitive> primitives;
//     primitives.push_back(mkldnn::softmax_forward(
//                            mkldnn::softmax_forward::primitive_desc(softmax_desc, _engine),
//                            mkldnn::primitive::at(input_memory),
//                            output_memory));
//     mkldnn::stream(mkldnn::stream::kind::eager).submit(primitives).wait();
//     return _output;
//   }

// private:
//   mkldnn::engine& _engine;
// };

std::vector<float>& embed(const Variable& var, const std::vector<unsigned int> ids) {
  static std::vector<float> emb;
  unsigned int depth = var.dim()[1];
  unsigned int batch_size = ids.size();
  emb.resize(batch_size * depth);

  const float* src = var.data();
  float* dst = emb.data();

  for (unsigned int i = 0; i < batch_size; ++i) {
    std::memcpy(dst + (i * depth), src + (ids[i] * depth), depth * sizeof (float));
  }

  return emb;
}

void concat(mkldnn::engine& engine,
            std::vector<std::vector<float> >& inputs,
            std::vector<float>& output,
            const std::vector<int>& dimension,
            int concat_dimension,
            mkldnn::memory::format format,
            mkldnn::memory::data_type data_type = mkldnn::memory::data_type::f32) {
  std::vector<mkldnn::memory::primitive_desc> inputs_primitive_desc;
  std::vector<mkldnn::memory> inputs_memory;
  std::vector<mkldnn::primitive::at> inputs_primitive_at;

  for (auto& input : inputs) {
    auto input_desc = mkldnn::memory::desc(dimension, data_type, format);
    inputs_primitive_desc.emplace_back(input_desc, engine);
    inputs_memory.emplace_back(inputs_primitive_desc.back(), input.data());
    inputs_primitive_at.emplace_back(inputs_memory.back(), 0);
  }

  std::vector<int> output_dimension(dimension);
  output_dimension[concat_dimension] *= inputs.size();
  int output_size = 1;
  for (int dim : output_dimension)
    output_size *= dim;
  output.resize(output_size);

  auto output_desc = mkldnn::memory::desc(output_dimension, data_type, format);
  auto output_memory = mkldnn::memory({output_desc, engine}, output.data());

  auto concat_desc = mkldnn::concat::primitive_desc(concat_dimension, inputs_primitive_desc);

  std::vector<mkldnn::primitive> primitives;
  primitives.push_back(mkldnn::concat(concat_desc, inputs_primitive_at, output_memory));
  mkldnn::stream(mkldnn::stream::kind::eager).submit(primitives).wait();
}

void test_concat() {
  std::vector<float> a = {1, 2, 3, 1, 2, 3};
  std::vector<float> b = {3, 4, 5, 6, 3, 4, 5, 6};
  std::vector<float> out(a.size() + b.size());
  //std::vector<std::vector<float> > in({a, b});
  //concat(engine, in, out, {2, 2}, 1, mkldnn::memory::format::nc);
  concat_in_depth({a.data(), b.data()}, {3, 4}, 2, out.data());

  for (auto v : out)
    std::cout << " " << v;
  std::cout << std::endl;
}

std::vector<float> build_scaled_time(unsigned int max_time, unsigned int depth) {
  // No need to optimize here, this is a one time operation.
  float log_timescale_increment = log(10000) / (depth - 1);
  std::vector<float> timescales(depth, -log_timescale_increment);
  for (unsigned int i = 0; i < timescales.size(); ++i)
    timescales[i] = exp(timescales[i] * i);

  std::vector<float> scaled_time(depth * max_time);
  for (unsigned int i = 0; i < max_time; ++i) {
    for (unsigned int j = 0; j < depth; ++j) {
      scaled_time[j + (i * depth)] = (i + 1) * timescales[j];
    }
  }

  return scaled_time;
}

std::vector<float> precompute_position_encoding(unsigned int max_time,
                                                unsigned int depth) {
  const std::vector<float> scaled_time = build_scaled_time(max_time, depth / 2);
  std::vector<float> sin_encoding(scaled_time.size());
  std::vector<float> cos_encoding(scaled_time.size());

  vsSin(scaled_time.size(), scaled_time.data(), sin_encoding.data());
  vsCos(scaled_time.size(), scaled_time.data(), cos_encoding.data());

  std::vector<float> position_encoding(sin_encoding.size() + cos_encoding.size());
  concat_in_depth({sin_encoding.data(), cos_encoding.data()},
                  {depth / 2, depth / 2}, max_time,
                  position_encoding.data());

  // std::vector<std::vector<float> > sin_cos_encoding = {sin_encoding, cos_encoding};
  // std::vector<int> dimensions = {max_time, depth / 2};
  // std::vector<float> encoding;
  // concat(engine, sin_cos_encoding, encoding, dimensions, 1, mkldnn::memory::format::nc);
  return position_encoding;
}

std::vector<float>& embed_and_encode(const Variable& embeddings,
                                     const std::vector<unsigned int>& ids,
                                     const std::vector<unsigned int>& lengths) {
  unsigned int batch_size = lengths.size();
  unsigned int depth = embeddings.dim()[1];

  static const std::vector<float> position_encoding = precompute_position_encoding(100, depth);
  std::vector<float>& embedded = embed(embeddings, ids);

  cblas_sscal(embedded.size(), 1.0 / sqrt(depth), embedded.data(), 1);

  unsigned int offset = 0;
  for (unsigned int i = 0; i < batch_size; ++i) {
    const int n = lengths[i] * depth;
    const float* x = position_encoding.data();
    float* y = embedded.data() + (offset * depth);
    cblas_saxpy(n, 1, x, 1, y, 1);
    offset += lengths[i];
  }

  return embedded;
}

void pad_sequence(const std::vector<float>& input,
                  unsigned int depth,
                  const std::vector<unsigned int>& lengths,
                  std::vector<float>& output) {
  unsigned int batch_size = lengths.size();
  unsigned int max_length = *std::max_element(lengths.begin(), lengths.end());
  output.resize(batch_size * max_length * depth);
  const float* src = input.data();
  float* dst = output.data();
  for (const auto length : lengths) {
    unsigned int count = depth * length;
    memcpy(dst, src, count * sizeof (float));
    dst += count;
    src += count;
    if (length < max_length) {
      count = (max_length - length) * depth;
      memset(dst, 0, count * sizeof (float));
      dst += count;
    }
  }
}

int main() {
  const char* model_path = "/home/klein/dev/ctransformer/model.bin";
  const char* vocab_path = "/home/klein/dev/OpenNMT-tf/models/averaged-ende-export500k/export/manual/1519808686/assets/wmt14-ende.vocab";

  void* model = load_model(model_path);
  if (model == nullptr)
    return 1;

  VariableIndex variable_index = build_variable_index(model);
  for (const auto& index : variable_index)
    std::cout << index.first << ": " << index.second << std::endl;

  Vocabulary vocabulary(vocab_path);

  std::vector<std::vector<std::string> > input = {
    {"▁Gut", "ach", ":", "▁Increase", "d", "▁safety", "▁for", "▁pedestrian", "s"},
    {"▁They", "▁are", "▁not", "▁even", "▁100", "▁metres", "▁apart"}
  };

  std::vector<unsigned int> ids;
  std::vector<unsigned int> lengths;
  unsigned int max_length = 0;
  unsigned int cum_length = 0;
  for (const auto& sentence : input) {
    unsigned int length = 0;
    for (const auto& token : sentence) {
      ids.push_back(vocabulary.to_id(token));
      ++length;
    }
    if (length > max_length)
      max_length = length;
    cum_length += length;
    lengths.push_back(length);
  }

  auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);

  const Variable& encoder_embeddings = get_variable(variable_index, "transformer/encoder/w_embs");
  std::vector<float>& encoder_input = embed_and_encode(encoder_embeddings, ids, lengths);

  unsigned int BT = cum_length;
  unsigned int B = lengths.size();
  unsigned int T = max_length;
  unsigned int D = 512;
  unsigned int H = 8;
  unsigned int DK = D / H;

  LayerNorm layer_norm(variable_index, "transformer/encoder/layer_0/multi_head/LayerNorm");
  std::vector<float> norm(BT * D);
  layer_norm.compute(encoder_input.data(), BT, D, norm.data());

  Dense dense(variable_index, "transformer/encoder/layer_0/multi_head/conv1d");
  std::vector<float> fused_proj(BT * dense.output_depth());
  dense.compute(norm.data(), BT, D, fused_proj.data());

  mkl_simatcopy('R', 'T', BT, D, 1.0, fused_proj.data(), D, BT);
  unsigned int chunk_size = BT * D;
  std::vector<float> queries(fused_proj.data(), fused_proj.data() + chunk_size);
  std::vector<float> keys(fused_proj.data() + chunk_size, fused_proj.data() + chunk_size * 2);
  std::vector<float> values(fused_proj.data() + chunk_size * 2, fused_proj.data() + chunk_size * 3);

  cblas_sscal(chunk_size, 1.0 / sqrt(D / H), queries.data(), 1);

  // D x B

  mkl_simatcopy('R', 'T', D, BT, 1.0, queries.data(), BT, D);
  mkl_simatcopy('R', 'T', D, BT, 1.0, keys.data(), BT, D);
  mkl_simatcopy('R', 'T', D, BT, 1.0, values.data(), BT, D);

  std::vector<float> padded_queries;
  std::vector<float> padded_keys;
  std::vector<float> padded_values;
  pad_sequence(queries, D, lengths, padded_queries);
  pad_sequence(keys, D, lengths, padded_keys);
  pad_sequence(values, D, lengths, padded_values);

  // b x T x D
  Eigen::TensorMap<Tensor4D> t_queries_map(padded_queries.data(), B, T, H, DK);
  Eigen::TensorMap<Tensor4D> t_keys_map(padded_keys.data(), B, T, H, DK);
  Eigen::TensorMap<Tensor4D> t_values_map(padded_values.data(), B, T, H, DK);

  Eigen::array<int, 4> shuffling;
  shuffling[0] = 0;
  shuffling[1] = 2;
  shuffling[2] = 1;
  shuffling[3] = 3;

  Tensor4D t_queries = t_queries_map.shuffle(shuffling);
  Tensor4D t_keys = t_keys_map.shuffle(shuffling);
  Tensor4D t_values = t_values_map.shuffle(shuffling);


  DotProductAttention attention(cpu_engine);
  std::vector<float> context(B * H * T * DK);
  attention(t_queries.data(),
            t_keys.data(),
            t_values.data(),
            B * H,
            T,
            T,
            DK,
            context.data());

  Eigen::TensorMap<Tensor4D> t_context(context.data(), B, H, T, DK);
  Tensor4D t_combined = t_queries_map.shuffle(shuffling);

  Dense dense_1(variable_index, "transformer/encoder/layer_0/multi_head/conv1d_1");
  std::vector<float> outputs(B * T * dense_1.output_depth());
  dense_1.compute(t_combined.data(), B * T, D, outputs.data());

  test_concat();

  return 0;
}

// class MultiHeadAttention
// {
// public:
//   MultiHeadAttention(bool is_self_attention,
//                      unsigned int num_heads,
//                      const std::string& scope,
//                      const VariableIndex& variable_index,
//                      mkldnn::engine& engine)
//     : _num_heads(num_heads),
//     , _scope(scope)
//     , _engine(engine)
//     , _layer_norm(variable_index, scope + "/LayerNorm") {
//     unsigned int num_kernels = is_self_attention ? 2 : 3;
//     for (unsigned int i = 0; i < num_kernels; ++i) {
//       std::string conv_scope = scope + "/conv1d";
//       if (i > 0) {
//         conv_scope += "_" + std::to_string(i);
//       }
//       _projections.emplace_back(variable_index, conv_scope);
//     }
//   }

//   void compute(const float* queries,
//                const float* memory,
//                unsigned int total_batch_size,
//                unsigned int queries_time,
//                unsigned int memory_time,
//                unsigned int depth;
//                float* output) {
//     maybe_resize(_queries_norm, total_batch_size * depth);
//     maybe_resize(_fused_proj, total_batch_size * depth * 3);

//     _layer_norm.apply(queries, total_batch_size, depth, _queries_norm);
//     _projections[0].compute(_queries_norm.data(), total_batch_size, depth, _fused_proj.data());

//     float* queries_proj;
//     float* keys_proj;
//     float* values_proj;

//     if (memory == nullptr) {

//     } else {
//       _projections[0].compute(memory.data(), total_batch_size, depth, _fused_proj.data());

//     }

//     std::vector<float> fused_proj(BT * dense.output_depth());
//   dense.compute(norm.data(), BT, D, fused_proj.data());

//   mkl_simatcopy('R', 'T', BT, D, 1.0, fused_proj.data(), D, BT);
//   unsigned int chunk_size = BT * D;
//   std::vector<float> queries(fused_proj.data(), fused_proj.data() + chunk_size);
//   std::vector<float> keys(fused_proj.data() + chunk_size, fused_proj.data() + chunk_size * 2);
//   std::vector<float> values(fused_proj.data() + chunk_size * 2, fused_proj.data() + chunk_size * 3);

//   cblas_sscal(chunk_size, 1.0 / sqrt(D / H), queries.data(), 1);

//   // D x B

//   mkl_simatcopy('R', 'T', D, BT, 1.0, queries.data(), BT, D);
//   mkl_simatcopy('R', 'T', D, BT, 1.0, keys.data(), BT, D);
//   mkl_simatcopy('R', 'T', D, BT, 1.0, values.data(), BT, D);

//   std::vector<float> padded_queries;
//   std::vector<float> padded_keys;
//   std::vector<float> padded_values;
//   pad_sequence(queries, D, lengths, padded_queries);
//   pad_sequence(keys, D, lengths, padded_keys);
//   pad_sequence(values, D, lengths, padded_values);

//   // b x T x D
//   Eigen::TensorMap<Tensor4D> t_queries_map(padded_queries.data(), B, T, H, DK);
//   Eigen::TensorMap<Tensor4D> t_keys_map(padded_keys.data(), B, T, H, DK);
//   Eigen::TensorMap<Tensor4D> t_values_map(padded_values.data(), B, T, H, DK);

//   Eigen::array<int, 4> shuffling;
//   shuffling[0] = 0;
//   shuffling[1] = 2;
//   shuffling[2] = 1;
//   shuffling[3] = 3;

//   Tensor4D t_queries = t_queries_map.shuffle(shuffling);
//   Tensor4D t_keys = t_keys_map.shuffle(shuffling);
//   Tensor4D t_values = t_values_map.shuffle(shuffling);


//   DotProductAttention attention(cpu_engine);
//   std::vector<float> context(B * H * T * DK);
//   attention(t_queries.data(),
//             t_keys.data(),
//             t_values.data(),
//             B * H,
//             T,
//             T,
//             DK,
//             context.data());

//   Eigen::TensorMap<Tensor4D> t_context(context.data(), B, H, T, DK);
//   Tensor4D t_combined = t_queries_map.shuffle(shuffling);

//   Dense dense_1(variable_index, "transformer/encoder/layer_0/multi_head/conv1d_1");
//   std::vector<float> outputs(B * T * dense_1.output_depth());
//   dense_1.compute(t_combined.data(), B * T, D, outputs.data());

//   }

// private:
//   unsigned int _num_heads;
//   const std::string& _scope;
//   mkldnn::engine& _engine;
//   LayerNorm _layer_norm;
//   std::vector<Dense> _projections;
// };
