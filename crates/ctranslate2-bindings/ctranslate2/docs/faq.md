# Frequently asked questions

## How does it relate to the original CTranslate project?

The original [CTranslate](https://github.com/OpenNMT/CTranslate) project shares a similar goal which is to provide a custom execution engine for OpenNMT models that is lightweight and fast. However, it has some limitations that were hard to overcome:

* a strong dependency on LuaTorch and OpenNMT-lua, which are now both deprecated in favor of other toolkits;
* a direct reliance on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), which introduces heavy templating and a limited GPU support.

CTranslate2 addresses these issues in several ways:

* the core implementation is framework agnostic, moving the framework specific logic to a model conversion step;
* the call to external libraries (Intel MKL, cuBLAS, etc.) occurs as late as possible in the execution to not rely on a library specific logic.

## Why and when should I use this implementation instead of PyTorch or TensorFlow?

Here are some scenarios where this project could be used:

* You want to accelarate Transformer models for production usage, especially on CPUs.
* You need to embed models in an existing C++ application without adding large dependencies.
* Your application requires custom threading and memory usage control.
* You want to reduce the model size on disk and/or memory.

However, you should probably **not** use this project when:

* You want to train custom architectures not covered by this project.
* You see no value in the [project key features](https://github.com/OpenNMT/CTranslate2#key-features).

## What are the known limitations?

The current approach only exports the weights from existing models and redefines the computation graph via the code. This implies a strong assumption of the graph architecture executed by the original framework.

## What are the future plans?

There are many ways to make this project better and even faster. See the [open issues](https://github.com/OpenNMT/CTranslate2/issues) for an overview of current and planned features.

## Do you provide a translation server?

The [OpenNMT-py REST server](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392) is able to serve CTranslate2 models. See the [code integration](https://github.com/OpenNMT/OpenNMT-py/commit/91d5d57142b9aa0a0859fbfa0dd94f301f56f879) to learn more.
