// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "ctranslate2/ops/flash-attention/flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96>(Flash_fwd_params &params, cudaStream_t stream);
