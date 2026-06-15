# AutoPM-260612 vs main — Comparative Feature Analysis

Repo: `/run/media/iaan/1TB-WD/Github/lingbot-map`
Generated: 2026-06-15

- `AutoPM-260612` head: `e377500` — established latest AutoPM work (realtime coverage / BIM product).
- `main` head: `ab0639f` — fast-forwarded to upstream Robbyant/lingbot-map latest (reconstruction library + new benchmark suite).
- Merge-base: `4cd9860`.

---

## 1. Divergence Overview

The branches share the merge-base `4cd9860` and have since evolved in **fully orthogonal directions** — no overlapping feature work, but two genuine conflict points (`.gitignore`, `README.md`).

| Metric | Value |
|---|---|
| Merge-base | `4cd9860` |
| AutoPM ahead of main (`main..AutoPM-260612`) | **19 commits** |
| main ahead of AutoPM (`AutoPM-260612..main`) | **11 commits** |
| AutoPM-side diff vs base (`main...AutoPM-260612`) | 117 files, +13,140 / −1 |
| main-side diff vs base (`AutoPM-260612...main`) | 101 files, +17,490 / −3 |

**Direction of each branch**

- **AutoPM-260612** = a **realtime coverage / BIM product** built *around* the reconstruction core: a FastAPI/WebRTC streaming server, GPU inference worker, multi-window registration, TSDF fusion, scan→BIM Sim(3) alignment, auto-placement + coverage analysis, browser viewers, BIM tooling, and several PRD/design docs. It treats `lingbot_map/` as a library and adds two new subpackages to it (`bim/`, `export/`).
- **main (upstream)** = the **reconstruction library plus a new evaluation/benchmark suite**: a full `benchmark/` harness (9 datasets, configs, evaluators, viewer, report generator), `preprocess/` (Oxford Spires + CUDA point-visibility), a `scripts/` memory profiler, and README/paper updates marking the benchmark as released.

### AutoPM-only commits (`main..AutoPM-260612`, 19)
```
e377500 AutoPM: 진행 요약 문서를 현 아키텍처로 갱신
e7de181 AutoPM: M4 후보 패널에 corroboration 수치 표시
b6facd4 AutoPM: smoke 자동배치 게이트 + StaleFixture setup/teardown
9eff5a2 AutoPM: ProjectAnalyze 캐시 생성 (analyze/ 7종)
0f2aa36 AutoPM: save work before branch switch
d0d6f22 glb-native: 실정점 디코드 + 노드별 객체 + 남은 한계 M1b/M2/M3
3087a31 auto-place: 표면 반경 보정 채점(M1) + 게이트 재캘리브레이션
90b49ce coverage: GLB-native 동적 모델 + 자동 배치(중력정렬) + viewer 수정
f2eb218 Phase A: PCA normals + LBP3 format + ShaderMaterial splat rendering
4726cc6 Densify point output: 30K→270K per-frame cap, 15→8mm voxel, depth median filter
68bdb67 Improve multi-window alignment: 48fr/window, 16 overlap, 4 max windows, ICP refinement
556f1b7 Phase 2: re-enable WebRTC streaming inference with busy-guard + 10fps throttle
709fcc6 Phase 3 ext: correspondence-based BIM alignment via umeyama (Shift+click picker)
892bb21 Phase 3 (MVP): BIM upload/list/serve endpoints + viewer overlay UI
eb15956 AutoPM Tier 3: multi-window inference + Procrustes registration on overlap frames
8d0022e AutoPM Tier 2: TSDF volumetric fusion + Marching Cubes mesh (LBM1 format)
fc2df8c AutoPM Tier 1: confidence-weighted voxel fusion + radius outlier filter
49720d3 AutoPM: save current realtime+demo work
8a87d4a docs: plan BIM-VAMS integration
```

### main-only commits (`AutoPM-260612..main`, 11)
```
ab0639f Mark additional datasets as complete in README
5e2fd69 Add eth3d/neural_rgbd/tat/tum/seven_scenes benchmark datasets
63fddb3 Update README.md
2225e63 add seven scenes and update paper
7d3dc4c Add VBR and DROID-W benchmark datasets
b330304 Update README.md
5fc8f88 Update README with new datasets for evaluation
d583a0f Update environment variable format in YAML config
8c75d43 update README
682302e add preprocess for oxford
6930c69 add evaluation benchmark
```

### Top-level tree set differences
- **Only on AutoPM:** `analyze/`, `docs/`, `models/`, `realtime/`, `tools/`
- **Only on main:** `benchmark/`, `preprocess/`, `scripts/`
- **Shared:** `assets/`, `demo.py`, `demo_render/`, `example/`, `gct_profile.py`, `LICENSE.txt`, `lingbot_map/`, `lingbot-map_paper.pdf`, `pyproject.toml`, `README.md`, `.gitignore`

---

## 2. What AutoPM Adds On Top of main

### 2.1 Realtime FastAPI / WebRTC server
- **`realtime/server.py`** (3,641 lines) — the product's spine. FastAPI app with WebRTC ingest (aiortc), video upload, model/BIM upload-list-serve endpoints, WebSocket point-cloud broadcast, and the coverage web. Comment notes the GPU worker auto-disables on OOM while the coverage web stays functional.
- **`realtime/inference_worker.py`** (990 lines) — background GPU inference: holds a rolling 518-wide frame buffer, runs `GCTStream.inference_streaming` on a sliding window every `interval_s`, broadcasts results. Imports core (`lingbot_map.models.gct_stream`, `utils.pose_enc`, `utils.geometry`). Single heavy load at startup (~5 s / ~11 GB peak).

### 2.2 Multi-window registration (Sim3 / Procrustes)
- **`realtime/registration.py`** (260 lines) — aligns successive inference windows into a common frame using overlapping camera centers as correspondences. `procrustes_align()` delegates to **`lingbot_map.utils.geometry.umeyama`** (Umeyama Sim3); commit `68bdb67` adds ICP refinement (48 fr/window, 16 overlap, 4 max windows).

### 2.3 TSDF volumetric fusion
- **`realtime/tsdf.py`** (247 lines) — pure numpy + scikit-image (no Open3D, since Open3D lacks Python 3.14 support). Builds a world-space TSDF grid, accumulates weighted SDF per voxel from per-frame depth, extracts a zero-isosurface via Marching Cubes, samples per-vertex color. Backs commit `8d0022e` (Tier 2, LBM1 format).

### 2.4 Scan-to-model alignment (`lingbot_map/bim/` — Umeyama Sim3) — added to the shared core
New subpackage `lingbot_map/bim/` (the only core change besides `export/`):
- **`alignment.py`** (163) — `Sim3` dataclass + `solve_sim3_umeyama` closed-form similarity from ≥3 correspondence pairs; `apply_pose_c2w` transforms c2w poses. (Umeyama PAMI 1991.)
- **`metadata.py`** (125) — `BimObject` / `BimModel` / `load_bim_metadata`.
- **`projection.py`** (132) — `project_object_to_frame`, `ProjectionResult` (BIM objects → camera frames).
- **`progress.py`** (231) — `aggregate_evidence`, `decide_status`, `infer_topology`, `build_progress_report` (per-object installation status).
- **`README.md`** (50), **`__init__.py`** (26).

### 2.5 Scene export (also added to the shared core)
- **`lingbot_map/export/scene_export.py`** (197) + `__init__.py` (4) — `export_scene` writes `scene_export.json` + sampled PLY for downstream BIM analysis.
- Wired into **`demo.py`** via the new `--export_dir` / `--no_viewer` flags (see §4).

### 2.6 Auto-placement + coverage analysis
- Auto-placement (gravity-aligned, surface-radius-corrected scoring) and coverage scoring live in `realtime/server.py` and the viewers; iterated across commits `90b49ce` (GLB-native dynamic model + auto-placement), `3087a31` (M1 surface-radius scoring + gate recalibration), `d0d6f22` (real-vertex decode + per-node objects), `e7de181` (M4 candidate-panel corroboration display).
- Densification: commit `4726cc6` (30K→270K per-frame cap, 8 mm voxel, depth median filter, stricter outlier).

### 2.7 Web viewers (browser UI)
- **`realtime/coverage.html`** (2,293) — main coverage viewer.
- **`realtime/viewer.html`** (1,054) — point-cloud / BIM overlay viewer (manual alignment, Shift+click correspondence picker, ShaderMaterial splat rendering with back-face cull / lambert from `f2eb218`).
- **`realtime/upload.html`** (235), **`realtime/index.html`** (231), **`realtime/audit.html`** (128), **`realtime/coverage_report.html`** (287).

### 2.8 BIM tooling (`tools/`)
- **`coverage_web_smoke.py`** (366) — the documented smoke test (server-up gate; commit `b6facd4` adds auto-placement gate + StaleFixture setup/teardown).
- **`pag_to_bim.py`** (181), **`synth_scene_from_bim.py`** (141), **`analyze_bim_progress.py`** (132), **`_smoke_test.py`** (138).

### 2.9 docs/ PRDs and design notes
- **`docs/model-coverage-webview-prd.md`** (947), **`docs/bim-vams-absorption-plan.md`** (341), **`docs/scan-to-model-improvement-prd.md`** (143), **`docs/glb-auto-mapping-review.md`** (116, the coordinate-convention reference cited in CLAUDE.md), **`docs/coverage-progress-summary.md`** (32).
- `analyze/` — 7 ProjectAnalyze cache files (structure/features/refactoring/etc.) from commit `9eff5a2`.

### 2.10 Model registry (`models/`)
- **`models/pipe_duct.glb`** (3.7 MB) + **`models/pipe_duct.model_manifest.json`** (57) — GLB-native registry entry: 454 objects (166 Pipe / 134 PipeFitting / 78 Duct / 76 DuctFitting), `up_axis: Z_UP`, `gltf_yup_to_zup`, GUID mapping ratio 1.0, proxy bounds.
- 11 `.rfa` Revit family files under `models/`.
- Committed fixture data: 64 `realtime/_received_frames/*.jpg`, plus `realtime/audit_results.json`, `cert.pem`, `key.pem` (these inflate the file count but are not source).

---

## 3. What main (upstream latest) Has That AutoPM Lacks

### 3.1 `benchmark/` — full evaluation suite (the headline addition; ~90 files)
- **Entry points:** `evaluate.py` (271), `run.py` (325), `run_worker.py` (185), `prepare.py` (251), `report.py` (57), `viewer.py` (2,000).
- **`benchmark/benchmark/` package:**
  - `core/` — `config.py` (265), `evaluator.py` (198), `loader.py` (549), `registry.py` (139), `saver.py` (433), `storage.py` (274).
  - `evaluation/` — `auc.py` (340), `depth.py` (359), `points.py` (540), `trajectory.py` (316).
  - `geometry/` — `projection.py`, `quaternion.py`, `registration.py` (380), `resize.py` (379), `transform.py`.
  - `io/` — `image.py`, `intrinsics.py`, `pointcloud.py`, `sampling.py`, `trajectory.py`.
  - `method/base.py` (143), `dataset/base.py` (273).
  - `report/` — `generator.py`, `manifest_builder.py` (368), HTML/JS/CSS templates (`app.js` 829, `styles.css` 477).
  - `utils/` — `logging.py`, `sky_segmentation.py`, `visualization.py`.
- **Dataset loaders (`benchmark/datasets/`):** `kitti.py`, `oxford_spires.py`, `tum.py`, `tnt.py`, `vbr.py`, `droid_w.py`, `eth3d.py`, `neural_rgbd.py`, `seven_scenes.py`, `general.py`.
- **Configs (`benchmark/configs/`):** per-dataset YAMLs (kitti, oxford(+long), tum, tat, vbr, droid_w, eth3d, neural_rgbd, seven_scenes) + dataset sub-configs + `methods/lingbot_map.yaml` and `methods/lingbot_map_v1.yaml` (these wrap the core checkpoint for evaluation).
- **Method wrapper:** `benchmark/methods/lingbot_map.py` (293) — `BaseMethod` subclass that imports the `lingbot_map` module.
- **Env installers:** `envs/install_all.sh`, `install_bench.sh`, `install_lingbot_map.sh`.
- **Assets/docs:** `benchmark/README.md` (636) + `README_zh.md` (669) + trajectory PNGs.

### 3.2 `preprocess/`
- `oxford.py` (838) — Oxford Spires preparation.
- `points_visibility/` — CUDA frustum-cull + visibility kernels (`frustum_cull.cu` 148, `visibility.cpp` 89, `visibility_kernel.cu` 493).

### 3.3 `scripts/`
- `benchmark_gct_memory.py` (418) — GCT memory profiler.

### 3.4 README / paper / assets updates
- `README.md` — News entry "Evaluation benchmark released", TODO list flips benchmark + 9 datasets to ✅, teaser swapped `assets/teaser.png` → `assets/teaser.webp`, VRAM-constrained fork link added.
- `lingbot-map_paper.pdf` updated (18,038,821 → 18,057,974 bytes).
- Removed `demo_render/.DS_Store`.

---

## 4. Shared Core — `lingbot_map/` and `demo.py`

`git diff --stat main AutoPM-260612 -- lingbot_map demo.py`:
```
 demo.py                            |  33 ++++++
 lingbot_map/bim/README.md          |  50 ++++++++
 lingbot_map/bim/__init__.py        |  26 +++++
 lingbot_map/bim/alignment.py       | 163 ++++++++++++++++++++++++++
 lingbot_map/bim/metadata.py        | 125 ++++++++++++++++++++
 lingbot_map/bim/progress.py        | 231 +++++++++++++++++++++++++++++++++++++
 lingbot_map/bim/projection.py      | 132 +++++++++++++++++++++
 lingbot_map/export/__init__.py     |   4 +
 lingbot_map/export/scene_export.py | 197 +++++++++++++++++++++++++++++++
 9 files changed, 961 insertions(+)
```

**Key finding: the existing ML core is byte-identical between the branches.** The shared subpackages `aggregator/`, `heads/`, `layers/`, `models/`, `utils/`, `vis/` show **zero diff**. AutoPM only **adds** two brand-new subpackages (`bim/`, `export/`) — these are net-new files, not modifications of upstream code.

`demo.py` differs by only **+33 lines, additive and opt-in**:
- New args `--export_dir` and `--no_viewer`.
- A guarded block that, when `--export_dir` is set, imports `lingbot_map.export.export_scene` and writes `scene_export.json` + PLY; `--no_viewer` returns early before the viser viewer.
- No existing `demo.py` line was changed — purely appended argparse options and a conditional branch.

**Implication:** main's benchmark suite imports the same untouched core modules AutoPM relies on; there is no source-level divergence in the reconstruction model itself.

---

## 5. Integration Risks / Notes (rebasing AutoPM onto latest main)

**Overall risk: low.** The two branches touch nearly disjoint file sets. AutoPM lives almost entirely in new top-level dirs (`realtime/`, `tools/`, `docs/`, `models/`, `analyze/`) plus two new `lingbot_map/` subpackages; main lives entirely in new top-level dirs (`benchmark/`, `preprocess/`, `scripts/`). A rebase of AutoPM onto `ab0639f` should apply cleanly except for two text-conflict files.

### Expected merge conflicts (only 2 source files both branches edited)
1. **`.gitignore`** — guaranteed conflict. Both branches rewrote the tail from the merge-base in different directions:
   - main removed `weights/`, `docs/`, `.DS_Store`, `**/.DS_Store`, `.claude-remote`.
   - AutoPM **kept `docs/` tracked** (it added real docs there) and added runtime ignores (`realtime/_uploads/`, `realtime/_received_frames/`, `realtime/_bim/`, `ckpts/`, `out/`, `video/`, `models/old/`).
   Resolution: union the two ignore sets, but do **not** re-add `docs/` (AutoPM intentionally tracks `docs/`).
2. **`README.md`** — main rewrote the News/TODO sections, swapped the teaser image, and added the benchmark/fork links. AutoPM did not touch README, so this is only a conflict if AutoPM later edits README; on a clean rebase it applies as a main-side change with no AutoPM counterpart. Low effort.

### Dependency direction
- **AutoPM → core:** AutoPM depends on existing core symbols that are present and unchanged on main:
  - `lingbot_map.utils.geometry.umeyama`, `closed_form_inverse_se3_general`, `unproject_depth_map_to_point_map`
  - `lingbot_map.models.gct_stream.GCTStream`
  - `lingbot_map.utils.pose_enc.pose_encoding_to_extri_intri`
  These all exist on `main` (the benchmark's `methods/lingbot_map.py` imports the same module), so AutoPM's imports remain valid after rebase.
- **`realtime/` is independent of `benchmark/`** — no cross-imports either way. They can coexist without interaction.
- **`demo.py`** — AutoPM's +33-line additive change does not collide with any main edit to `demo.py` (main did not modify `demo.py`); reapplies cleanly.

### Things to bring over / watch
- After rebase, AutoPM gains the `benchmark/`, `preprocess/`, `scripts/` trees for free (no AutoPM code references them).
- The benchmark's `configs/methods/lingbot_map.yaml` points at a checkpoint path and `env: lingbot-map`; unrelated to the realtime stack — no action needed for AutoPM.
- Committed runtime/fixture blobs on AutoPM (`realtime/_received_frames/*.jpg`, `cert.pem`, `key.pem`, `audit_results.json`, `.rfa` files, `pipe_duct.glb`) are large but are real fixtures (CLAUDE.md marks the smoke uploads as do-not-modify). They do not conflict with main; just noted for repo-size awareness.

### Bottom line
A rebase is mechanically straightforward: resolve `.gitignore` by union (keep `docs/` tracked), take main's `README.md`, and all of AutoPM's net-new realtime/BIM stack reapplies unchanged because the reconstruction core it depends on is identical on both branches.
