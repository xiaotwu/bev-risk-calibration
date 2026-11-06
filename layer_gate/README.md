# Layer-Gate source overlay

This overlay preserves the original scene-conditioned Layer-Gate fusion implementation and its CPU source tests. It is a companion to the manuscript's project background. The stale-observation risk analysis has its own implementation and tests.

## Apply to the pinned base

Obtain the BEVFormerFusion checkout and apply the patch to a clean detached checkout of the recorded base. Replace the patch path with its absolute location in this repository. [Upstream repository](https://github.com/xiaotwu/BEVFormerFusion/tree/3b84adad901a6d24075bc1c05cd51a0f316c4187)

```sh
git clone https://github.com/xiaotwu/BEVFormerFusion.git
git -C BEVFormerFusion checkout --detach 3b84adad901a6d24075bc1c05cd51a0f316c4187
git -C BEVFormerFusion apply --check --unidiff-zero /absolute/path/to/layer_gate/layer-gate.patch
git -C BEVFormerFusion apply --unidiff-zero /absolute/path/to/layer_gate/layer-gate.patch
```

The patch changes four tracked files and creates nine files. The added files are also browsable under `overlay/`. Apply the patch once. `source_manifest.json` records the upstream revision and SHA-256 digests of the patch and all 13 resulting source files.

## Run the CPU source tests

Run the tests from the patched BEVFormerFusion root in a Python environment with NumPy and PyTorch.

```sh
python -B -m unittest discover -s tests/layergate
```

The tests cover source-level fusion routing, checkpoint conversion, and the gate-record contract. Full detector inference uses the upstream environment, a compatible CUDA stack, nuScenes data, and model weights. Follow the upstream setup and data terms for that workflow.
