# Local Chestnut builds

This fork supports explicitly registered, locally compiled tinygrad supercombo
models. This is not a compatibility or road-safety certification. A successful
warmup or synthetic benchmark does not validate live-camera driving behavior.

Keep assistance disabled while registering and testing. Model pickles execute
code: register only artifacts built from trusted source with a trusted compiler.
Hashes detect changes; they do not establish that a model is safe or trustworthy.

Put the artifact in `Paths.model_root()` with a unique `local-*.pkl` basename.
Chunked artifacts need the canonical `.chunkNNofMM` files and a `.chunkmanifest`
containing the chunk count. Do not retain both flat and chunked representations.
Put a JSON manifest in the `local/` subdirectory of the same model root, using
this structure (replace placeholders with actual pinned values):

```json
{
  "version": 1,
  "source": "chestnut",
  "target_arch": "gfx1102",
  "source_onnx_sha256": "<64 lowercase hex characters>",
  "compiler_commit": "<40 lowercase hex characters>",
  "bundle": {
    "index": 0,
    "internalName": "local-my-build",
    "displayName": "My local build (gfx1102, experimental)",
    "generation": 12,
    "runner": "tinygrad",
    "minimumSelectorVersion": 19,
    "is20hz": true,
    "ref": "local:<combined artifact SHA256>",
    "overrides": [{"key": "folder", "value": "Local experimental builds"}],
    "models": [{
      "type": "supercombo",
      "artifact": {
        "fileName": "local-my-build.pkl",
        "downloadUri": {
          "uri": "local://<combined artifact SHA256>",
          "sha256": "<combined artifact SHA256>"
        },
        "chunks": []
      }
    }]
  }
}
```

For chunked artifacts, populate `chunks` with `fileName` and `sha256` for every
chunk, in order. The combined hash covers all chunk bytes concatenated in order.
Generation, frame rate and overrides must match the source model and compilation;
do not copy example values without checking. Preserve source ref, compiler and
tinygrad commits, build flags and test reports alongside your private build data.

Registration is **not selection**. Valid local builds appear in the Chestnut
catalog, including the cached/offline UI path, without changing either selected
slot. The loader checks the detected AMD architecture before unpickling. Remote
catalog refreshes cannot replace a local build with a different architecture.
Missing or changed artifacts invalidate the local selection rather than download
a replacement. Verification hashes are cached per process only while manifest
and artifact stat identities remain unchanged. Ordinary cache clearing preserves
`local-` files and the registration directory, including unselected builds;
remove local builds explicitly when no longer needed.

Model initialization and warmup run at normal scheduling priority. FIFO priority
is applied only immediately before the modeld frame loop. Loader threads drop
inherited real-time priority explicitly. Do not disable RT throttling/watchdogs
to work around initialization stalls.
