# Memory management

## From Python

To release the resources used by [`Translator`](python/ctranslate2.Translator.rst) and [`Generator`](python/ctranslate2.Generator.rst) instances, you can simply delete the object, e.g.:

```python
del translator
```

In some cases, you might want to temporarily unload the model and load it back later. The `Translator` object provides the methods `unload_model` and `load_model` for this purpose. The model can be either fully unloaded or moved to the CPU memory.
