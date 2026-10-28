# Checkpoints

`checkpoints.csv` lists the benchmark best/last files by original run path,
SHA-256 and byte size. The `.pt` files are supplied separately to avoid including
large duplicated training states in the code archive.

For evaluation, use the **best** checkpoint selected on validation, not the last
checkpoint or the highest-scoring test run. Keep its `protocol.json`, environment
record and exact architecture source together. The source hash and radiometry
checks in the evaluator must not be disabled.

Load only checkpoints obtained from a trusted source: the historical training
states use Python pickle serialization. Do not edit or re-save an original
checkpoint merely to make its protocol match different code.
