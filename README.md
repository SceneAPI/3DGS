# 3dgs

One `sfmapi-plugin-http-v1` container-service package for the five radiance-field
training providers that previously shipped as five separate plugin repos. Each
provider keeps its own manifest, entry point, and launcher; the four
near-identical native-engine trainers are unified into a single engine module
with a per-provider config table, and gsplat's in-process CUDA trainer stays
its own module.

| Provider id | Display name | Engine | Trainer module | Default service port |
|---|---|---|---|---|
| `brush` | Brush | native wgpu/Vulkan build (`/opt/brush`) | `gs3.trainer` | 8096 |
| `gsplat` | gsplat | in-process CUDA torch + gsplat | `gs3.gsplat_trainer` | 8098 |
| `fastergs` | Faster-GS | NeRFICG Faster-GS checkout (`/opt/fastergs`) | `gs3.trainer` | 8093 |
| `lfs` | LichtFeld Studio | native CUDA build (`/opt/LichtFeld-Studio`) | `gs3.trainer` | 8095 |
| `spirulae` | spirulae-splat | spirulae-splat checkout (`/opt/spirulae-splat`) | `gs3.trainer` | 8094 |

Every service exposes `/healthz`, `/version`, `/capabilities`, and `/execute`
through the `sceneapi.plugin_service` kit at protocol `sfmapi-plugin-http-v1`
version 1.1, and dispatches `radiance_train` / `radiance_eval` tasks to its
provider's trainer.

## Install and extras

Base install covers brush / lfs / spirulae / fastergs — their engines are
native builds driven over subprocess (torch for spirulae/fastergs arrives via
the container image, as before):

```bash
uv pip install 3dgs
```

Extras:

- `gsplat` — pillow, numpy, pycolmap (gsplat's non-CUDA runtime deps; they
  were hard deps of the old `sfmapi-gsplat` package).
- `gsplat-cuda` — the above plus `torch==2.7.1`, `gsplat==1.5.3`,
  `lpips>=0.1.4` (the old `sfmapi-gsplat[cuda]` extra).
- `test` — pytest, ruff, httpx, plus the gsplat import deps so the whole
  suite runs without CUDA.

## Running a provider service

The five launchers keep their old names and defaults (`0.0.0.0:8080`) — they
are deployment-facing (container CMDs, operator launchers), so the 0.1.0
package rename intentionally does NOT rename them and adds no
sceneapi-prefixed aliases:

```bash
sfmapi-brush --host 127.0.0.1 --port 8096
sfmapi-gsplat --host 127.0.0.1 --port 8098
sfmapi-fastergs --host 127.0.0.1 --port 8093
sfmapi-lfs --host 127.0.0.1 --port 8095
sfmapi-spirulae --host 127.0.0.1 --port 8094
```

Programmatically: `from gs3.server import build_app;
app = build_app("brush")` (replaces the old `sfmapi_brush.server:app`
module attribute).

## Development

```bash
uv venv
uv sync --extra test
uv run ruff check .
uv run pytest -q
```

`sceneapi` resolves from the sibling core checkout (`../sfmapi`) via
`[tool.uv.sources]`; CI checks out `SceneAPI/SceneAPI` into `.deps/sceneapi`
and installs this package `--no-deps`.

## Rename (0.1.0): sfmapi-radiance → 3dgs

This package was renamed from `sfmapi-radiance` (import package
`sfmapi_radiance`) as part of the sfmapi → sceneapi migration; the repo moved
to `SceneAPI/3DGS`. What changed and what deliberately did not:

- **Distribution / import package**: `3dgs` / `gs3`.
- **Entry-point group**: manifests register under `sceneapi.backends` with
  UNCHANGED provider names (`brush`, `gsplat`, `fastergs`, `lfs`,
  `spirulae`); the core still reads the legacy `sfmapi.backends` group for
  one release.
- **Console scripts**: unchanged (`sfmapi-brush`, `sfmapi-gsplat`,
  `sfmapi-fastergs`, `sfmapi-lfs`, `sfmapi-spirulae`) — kept verbatim this
  release, no aliases added.
- **Plugin-owned `SFMAPI_*` env names are kept** — they are manifest-owned
  container contract, not core settings: `SFMAPI_<PROVIDER>_SERVICE_URL`,
  `SFMAPI_PLUGIN_OBJECT_STORE_URL`, `SFMAPI_PLUGIN_OUTPUT_ROOT`,
  `SFMAPI_PLUGIN_WORK_ROOT`, `SFMAPI_PLUGIN_EXECUTE_TIMEOUT`,
  `SFMAPI_<PROVIDER>_ROOT`, `SFMAPI_BRUSH_EXECUTABLE`,
  `SFMAPI_LFS_EXECUTABLE`, `SFMAPI_GSPLAT_OUTPUT_ROOT`,
  `SFMAPI_FASTERGS_FRAMEWORK_ROOT`, and the `SFMAPI_*_REF` Docker engine
  build args.
- **Wire identity unchanged** (until migration Phase C): protocol
  `sfmapi-plugin-http-v1`, artifact formats `sfmapi.radiance.*.v1`, the
  `/sfmapi/{input,output,work,logs,cache}` container mount contract, and the
  manifests' `compatibility.sfmapi` field name.

## Migration (decision D3)

This package supersedes the five per-provider repos — `sfmapi_brush`,
`sfmapi_gsplat`, `sfmapi_fastergs`, `sfmapi_lfs`, `sfmapi_spirulae` — which
are to be archived by their owner.

- **Entry points are unchanged**: the backend entry-point group still exposes
  `brush`, `gsplat`, `fastergs`, `lfs`, and `spirulae`, so a deployment swaps
  packages without config changes. Only the entry-point *values* move (e.g.
  `sfmapi_brush.plugin:plugin` → `gs3.providers.brush:plugin`).
- **Console scripts are unchanged**: `sfmapi-brush`, `sfmapi-gsplat`,
  `sfmapi-fastergs`, `sfmapi-lfs`, `sfmapi-spirulae`.
- **Manifest identity now names this repo.** Each manifest's identity
  fields — `package_name` (`3dgs`), `github_url`
  (`SceneAPI/3DGS`), `entry_points`
  (`gs3.providers.<provider>:plugin`), the `uv` install
  coordinates, and the `docker` / `container_service.image.build` contexts —
  point at this repo. The five per-provider Dockerfiles were ported to
  `docker/<provider>.Dockerfile` (engine build steps unchanged; only the
  plugin-package install + CMD moved to this package), and
  `container_service.image.build.dockerfile` selects the right one per
  provider. Engine checkouts inside the Dockerfiles (ArthurBrussee/brush,
  MrNeRF/LichtFeld-Studio, nerficg-project, harry7557558/spirulae-splat) and
  their `SFMAPI_*_REF` build args are untouched — those are external
  upstreams, not plugin coordinates. Note the plain `docker` runtime mode's
  `build_context` has no per-provider Dockerfile selector (the schema has no
  `dockerfile` field there), so a bare `docker build` of the repo root finds
  no Dockerfile; use the container_service build (or
  `docker build -f docker/<provider>.Dockerfile`) instead.
- **Direct importers**: `sfmapi_<pkg>.plugin` → `gs3.providers.<provider>`;
  `sfmapi_<pkg>.trainer` → `gs3.trainer` (brush/lfs/spirulae/fastergs,
  now parameterized by `request.provider`) or `gs3.gsplat_trainer`;
  `sfmapi_<pkg>.server:app` → `gs3.server:build_app(<provider>)`.
- `sfmapi-gsplat[cuda]` → `3dgs[gsplat-cuda]`.
