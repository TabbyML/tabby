# Marian

CTranslate2 supports Transformer models trained with [Marian](https://github.com/marian-nmt/marian). The conversion requires the paths to the model and vocabularies:

```bash
ct2-marian-converter --model_path model.npz --vocab_paths src.vocab.yml tgt.vocab.yml \
    --output_dir ct2_model
```

```{attention}
Models already converted with `marian-conv` are not supported.
```
