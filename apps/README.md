# Service 4 incident explorer

The explorer is a read-only interface over precomputed state-machine v1.0, v1.1 and v2.0 parent
artifacts. Moving the time slider or changing a filter never reconstructs episodes. Asset choices
show the immutable numeric ID together with a stable location label; filtering still uses the
numeric ID.

From the Service 4 project directory:

```bash
uv pip install --python ../.venv/bin/python -e '.[viewer]'
../.venv/bin/service4-build-v1
../.venv/bin/service4-build-v1-1
../.venv/bin/service4-build-v2
../.venv/bin/streamlit run apps/incident_explorer.py
```

Each builder verifies the enrichment handoff and writes checksum-recorded outputs to its versioned
directory. If that directory already contains different bytes, it stops instead of overwriting the
artifact. Independent `v1.0`, `v1.1` and `v2.0` checkboxes are all enabled by default. Each selected
model is drawn on a separate episode lane over the same event timeline. The v2.0 lane shows frozen
parent episodes, not the separate diagnostic child phases.

The `Interesting sample` dropdown provides deterministic jump targets for control-confirmed
restoration, the largest v1.0/v2.0 boundary disagreement, v2.0 provisional restoration, two-stage
retry behavior, unresolved breakers, and one-event breaker/protection cases. A sample selects its
asset, the corresponding focus episode in the primary model, and the initial time window. Choose
`Browse manually` to use the asset and episode controls directly.

The initial viewer is deliberately read-only. It displays episode intervals, automatic event
members, linked control context, optional unassigned observations, unplanned-outage references,
announced work and the optional bounded-look-ahead evidence status. IZPADI and IZKLOPI use
separate labelled lanes; each visible interval is annotated with its source identifier. In the v1.1
view, a nonzero IZPADI interval receives a one-minute display extension because its source is only
minute-precise; a zero-duration reference remains a point and is drawn as a diamond. IZKLOPI remain
announced intent, not evidence that switching occurred. These are display annotations only: the
app does not write reviewer annotations or modify processed inputs.
