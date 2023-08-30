# Frequently Asked Questions

<details>
  <summary>How much VRAM a LLM model consumes?</summary>
  <div>By default, Tabby operates in int8 mode with CUDA, requiring approximately 8GB of VRAM for CodeLlama-7B.</div>
</details>

<details>
  <summary>What GPUs are required for reduced-precision inference (e.g int8)?</summary>
  <div>
    <ul>
      <li>int8: Compute Capability >= 7.0 or Compute Capability 6.1</li>
      <li>float16: Compute Capability >= 7.0</li>
      <li>bfloat16: Compute Capability >= 8.0</li>
    </ul>
    <p>
      To determine the mapping between the GPU card type and its compute capability, please visit <a href="https://developer.nvidia.com/cuda-gpus">this page</a>
    </p>
  </div>
</details>
