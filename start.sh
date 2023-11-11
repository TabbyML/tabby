docker run -it \
  -p 8080:8080 \
  -v $HOME/ai/apps/tabby-dev:/data \
  --gpus all \
  -e RUST_BACKTRACE=full \
  --device /dev/dri \
  tabbyml/tabby:opencl \
  serve \
  --model TabbyML/StarCoder-1B
  #--model TabbyML/CodeLlama-7B --chat-model TabbyML/WizardCoder-3B 

