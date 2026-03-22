# Metal Training Backend Design

## Goal

Add a real Metal training backend to RustGS so projection, rasterization, loss,
backward, and optimizer updates can execute on GPU instead of round-tripping
through the current hybrid CPU path.

## Why a New Backend

The existing `CompleteTrainer` is still a hybrid pipeline:

- projection runs on Candle/Metal
- projected tensors are copied back to CPU
- tiled rasterization and analytical backward run on CPU
- gradients are uploaded again for GPU Adam

That makes Metal visible in logs, but it does not produce a true GPU training
loop. Extending that path incrementally would keep CPU ownership of the hottest
part of the pipeline.

## Chosen Architecture

Use an explicit backend switch:

- `legacy-hybrid`: current `CompleteTrainer`
- `metal`: new `MetalTrainer`

The new `MetalTrainer` keeps trainable Gaussian parameters in `Var`s on the
Metal device and performs:

1. GPU projection with Candle tensor ops
2. GPU chunked alpha compositing against a low-resolution training grid
3. GPU loss computation
4. `loss.backward()` on GPU
5. GPU Adam updates without CPU gradient uploads

## Initial Scope

The first Metal backend uses a chunked, low-resolution differentiable renderer.
This is a deliberate stepping stone:

- it is a true GPU training loop
- it avoids the current CPU rasterizer entirely
- it keeps implementation inside Candle/Metal tensor ops
- it creates a clean boundary for later replacement with native Metal kernels

## Next Phases

1. Replace chunked tensor rasterization with native Metal tile kernels.
2. Add GPU-side culling and tile list construction.
3. Add GPU depth/SSIM losses and densify/prune logic.
4. Promote `metal` to the default backend once parity and stability are good.
