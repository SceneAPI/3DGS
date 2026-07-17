# sfmapi-radiance

One `sfmapi-plugin-http-v1` container-service package for the five radiance-field
training providers that previously shipped as five separate plugin repos. Each
provider keeps its own manifest, entry point, and launcher; the four
near-identical native-engine trainers are unified into a single engine module
with a per-provider config table, and gsplat's in-process CUDA trainer stays
its own module.

| Provider id | Display name | Engine | Trainer module | Default service port |
|---|---|---|---|---|
| `brush` | Brush | native wgpu/Vulkan build (`/opt/brush`) | `sfmapi_radiance.trainer` | 8096 |
| `gsplat` | gsplat | in-process CUDA torch + gsplat | `sfmapi_radiance.gsplat_trainer` | 8098 |
| `fastergs` | Faster-GS | NeRFICG Faster-GS checkout (`/opt/fastergs`) | `sfmapi_radiance.trainer` | 8093 |
| `lfs` | LichtFeld Studio | native CUDA build (`/opt/LichtFeld-Studio`) | `sfmapi_radiance.trainer` | 8095 |
| `spirulae` | spirulae-splat | spirulae-splat checkout (`/opt/spirulae-splat`) | `sfmapi_radiance.trainer` | 8094 |

Every service exposes `/healthz`, `/version`, `/capabilities`, and `/execute`
through the `sfmapi.plugin_service` kit at protocol `sfmapi-plugin-http-v1`
version 1.1, and dispatches `radiance_train` / `radiance_eval` tasks to its
provider's trainer.

## Install and extras

Base install covers brush / lfs / spirulae / fastergs — their engines are
native builds driven over subprocess (torch for spirulae/fastergs arrives via
the container image, as before):

```bash
uv pip install sfmapi-radiance
```

Extras:

- `gsplat` — pillow, numpy, pycolmap (gsplat's non-CUDA runtime deps; they
  were hard deps of the old `sfmapi-gsplat` package).
- `gsplat-cuda` — the above plus `torch==2.7.1`, `gsplat==1.5.3`,
  `lpips>=0.1.4` (the old `sfmapi-gsplat[cuda]` extra).
- `test` — pytest, ruff, httpx, plus the gsplat import deps so the whole
  suite runs without CUDA.

## Running a provider service

The five launchers keep their old names and defaults (`0.0.0.0:8080`):

```bash
sfmapi-brush --host 127.0.0.1 --port 8096
sfmapi-gsplat --host 127.0.0.1 --port 8098
sfmapi-fastergs --host 127.0.0.1 --port 8093
sfmapi-lfs --host 127.0.0.1 --port 8095
sfmapi-spirulae --host 127.0.0.1 --port 8094
```

Programmatically: `from sfmapi_radiance.server import build_app;
app = build_app("brush")` (replaces the old `sfmapi_brush.server:app`
module attribute).

## Development

```bash
uv venv
uv sync --extra test
uv run ruff check .
uv run pytest -q
```

`sfmapi` resolves from the sibling checkout (`../sfmapi`) via
`[tool.uv.sources]`; CI checks out `SFMAPI/sfmapi` into `.deps/sfmapi` and
installs this package `--no-deps`.

## Migration (decision D3)

This package supersedes the five per-provider repos — `sfmapi_brush`,
`sfmapi_gsplat`, `sfmapi_fastergs`, `sfmapi_lfs`, `sfmapi_spirulae` — which
are to be archived by their owner.

- **Entry points are unchanged**: the `sfmapi.backends` group still exposes
  `brush`, `gsplat`, `fastergs`, `lfs`, and `spirulae`, so a deployment swaps
  packages without config changes. Only the entry-point *values* move (e.g.
  `sfmapi_brush.plugin:plugin` → `sfmapi_radiance.providers.brush:plugin`).
- **Console scripts are unchanged**: `sfmapi-brush`, `sfmapi-gsplat`,
  `sfmapi-fastergs`, `sfmapi-lfs`, `sfmapi-spirulae`.
- **Manifests are carried verbatim** (they were just synced to the core
  vocabulary at protocol 1.1). That deliberately includes each manifest's
  identity fields — `package_name`, `github_url`, `entry_points`, and the
  `container_service.image.build` context — which still name the original
  per-repo coordinates; container images therefore still build from the old
  repos' Dockerfiles. Re-pointing manifest identity at this repo is a
  follow-up decision.
- **Direct importers**: `sfmapi_<pkg>.plugin` → `sfmapi_radiance.providers.<provider>`;
  `sfmapi_<pkg>.trainer` → `sfmapi_radiance.trainer` (brush/lfs/spirulae/fastergs,
  now parameterized by `request.provider`) or `sfmapi_radiance.gsplat_trainer`;
  `sfmapi_<pkg>.server:app` → `sfmapi_radiance.server:build_app(<provider>)`.
- `sfmapi-gsplat[cuda]` → `sfmapi-radiance[gsplat-cuda]`.
