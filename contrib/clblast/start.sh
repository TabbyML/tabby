docker run -it \
  -p 8080:8080 \
  -v $HOME/ai/apps/tabby/.tabby:/data \
  --gpus all \
  -e RUST_BACKTRACE=1 \
  --device /dev/dri \
  tabbyml/tabby:clblast \
  serve \
  --model TabbyML/StarCoder-1B
  #--model TabbyML/CodeLlama-7B --chat-model TabbyML/WizardCoder-3B 
