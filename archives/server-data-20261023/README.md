# Full server experiment snapshot — server-data-20261023

Status: **COMPLETE — all assets verified**.

Source: `/root/autodl-tmp/layer-probe` on the experiment server. Snapshot: 15,049 regular files, 180,702,908,040 logical bytes, plus recorded directory and symbolic-link metadata. Excluded credentials/special files: 0.

Data travel directly from the experiment server to GitHub Releases. The user's computer is not a data relay. Code, prompts, generated answers, raw results, activations, adapter checkpoints, model weights, and the existing project environment/cache are included. Original source files are not removed or modified.

Download the release assets and use `restore.py` to verify and reconstruct the directory. Large files are losslessly gzip-compressed in independent 512-MiB input parts; the manifest maps every part to its original path, offset and SHA256. Small files are stored in a separate gzip-compressed tar archive. Every uploaded asset is verified against GitHub's SHA256 (or read back on the server if unavailable).

Release: https://github.com/Liushengyu258/frozen-exit-adapters/releases/tag/server-data-20261023

```bash
python3 restore.py --destination ./layer-probe-restored
```

This is an archival experiment snapshot, not a new evaluation or license grant. Original model/data licenses continue to apply; recorded environment symlinks may require the corresponding system interpreter to exist.
