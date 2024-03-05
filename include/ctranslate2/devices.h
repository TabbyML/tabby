#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#ifdef CT2_WITH_TENSOR_PARALLEL
#  include <nccl.h>
#endif

namespace ctranslate2 {

  enum class Device {
    CPU,
    CUDA
  };

  Device str_to_device(const std::string& device);
  std::string device_to_str(Device device);
  std::string device_to_str(Device device, int index);

  int get_device_count(Device device);

  int get_device_index(Device device);
  void set_device_index(Device device, int index);

  void synchronize_device(Device device, int index);
  void synchronize_stream(Device device);

  class ScopedDeviceSetter {
  public:
    ScopedDeviceSetter(Device device, int index)
      : _device(device)
      , _prev_index(get_device_index(device))
      , _new_index(index)
    {
      if (_prev_index != _new_index)
        set_device_index(device, _new_index);
    }

    ~ScopedDeviceSetter() {
      // Set previous device index.
      if (_prev_index != _new_index)
        set_device_index(_device, _prev_index);
    }

  private:
    Device _device;
    int _prev_index;
    int _new_index;
  };

  extern int my_rank;
  extern int local_rank;
  extern int n_ranks;

  class ScopedMPISetter {
  public:
    ScopedMPISetter();
    ~ScopedMPISetter();

    static int getNRanks();
    static int getCurRank();
    static int getLocalRank();

#ifdef CT2_WITH_TENSOR_PARALLEL
    static ncclComm_t getNcclComm();
#endif

    static void finalize();

  private:
#ifdef CT2_WITH_TENSOR_PARALLEL
    static uint64_t getHostHash(const char *string);
    static void getHostName(char *hostname, int maxlen);
    static std::vector<ncclComm_t*> _nccl_comms;
#endif
  };
}
