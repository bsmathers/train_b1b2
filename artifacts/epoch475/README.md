# TaurosV1 Epoch475 checkpoint

`policy_epoch_475.pt` is split into three ordinary Git files so every object is
below GitHub's 100 MB per-file limit. The CUDA launcher reconstructs it
automatically with `tools/assemble_epoch475.py` when the checkpoint is absent.

Combined checkpoint:

- Bytes: `139252367`
- SHA-256: `f7a8f40575127b36350b92f78f01f7a1c517ab4b9878a505fd4269baef844e54`

Shard checksums:

- `part-000`: `58e80ce0c2f5968775556c3c0a681c67909a04ab8898b0fb805cf68ca6fef52a`
- `part-001`: `82b5f34477794aee52b8f63b15af3a45e4940cbb2aa653ce00419f78cb8c6750`
- `part-002`: `e470cf210ffe687e63e78a7ea7df294ed8c6127c48c12445bab3f055bac16a11`

Manual reconstruction:

```bash
python tools/assemble_epoch475.py --repo-root .
```
