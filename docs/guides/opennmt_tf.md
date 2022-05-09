# OpenNMT-tf

CTranslate2 supports Transformer models trained with [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf). In most cases, you should simply pass the YAML configuration to the converter:

```bash
pip install OpenNMT-tf
ct2-opennmt-tf-converter --config config.yml --output_dir ct2_model
```

```{tip}
If you don't have access to the configuration or want to select a checkpoint outside the model directory, see the other conversion options with `ct2-opennmt-tf-converter -h`.
```

Alternatively, you can also convert the model directly from OpenNMT-tf. See the corresponding [documentation](https://opennmt.net/OpenNMT-tf/serving.html#ctranslate2) in OpenNMT-tf.
