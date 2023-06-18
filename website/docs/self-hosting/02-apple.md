# Mac M1/M2 (Preview)

Thanks to Apple's Accelerate and CoreML frameworks, we can now run Tabby on edge devices with reasonable inference speed. Follow the steps below to set it up:

1. Download the tabby binary from the latest Release page, rename it to `tabby`, place it in a directory included in your `$PATH` variable, and ensure its permissions are set to executable (e.g., 755).
3. Run `tabby --help` to verify successful installation.

3. Start the server with:
```bash
tabby serve --model TabbyML/T5P-220M
```


:::tip
The compute power of M1/M2 is limited and is likely to be sufficient only for individual usage. If you require a shared instance for a team, we recommend considering Docker hosting with CUDA. You can find more information about Docker [here](./docker).
