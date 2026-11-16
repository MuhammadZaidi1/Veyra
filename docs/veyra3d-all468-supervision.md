# All-468 supervision decision

The audit returns zero eligible points. This is conservative, but necessary:
geometry alone does not prove semantic correspondence. The loss is able to accept
a partial mask. Ten tests pass. The existing loader appropriately rejects an empty
mask. All old results are preserved, as are the model and topology.

The $0 alternative is to use the canonical mesh, which has exact identity by
construction. It does not support identity generalization. The next task is a
small canonical-only control rather than an automatic training run. This cannot
resolve the source anatomy, nor replace independent measurements.
