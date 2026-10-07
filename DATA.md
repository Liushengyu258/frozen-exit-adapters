# Data provenance and redistribution

The repository distributes derived numerical measurements and paired question IDs, not the source question/solution text, full model answers, activations or model weights.

Source datasets:
- GSM8K: https://github.com/openai/grade-school-math
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- Code Alpaca: https://github.com/sahil280114/codealpaca
- Model: https://huggingface.co/Qwen/Qwen3.8-27B

Consult upstream terms before downloading or redistributing those artifacts. The downloader validates original source file hashes in main/data_manifest.json and fails if current upstream bytes differ. Archived split manifests contain the preparation seed and output hashes. Immutable model revision is not known.

The published generation tables preserve IDs, domain, exact recorded pass/fail, reason categories, length and diagnostic measurements; they omit prompt, reference, answer, tests and machine timing. Their SHA-256 differs from the private full-output archive by design. Numeric reproductions do not rerun the teacher or judge.

Code is publicly available for inspection. An explicit reuse license has not yet been selected by the author; no license for upstream material is implied.
