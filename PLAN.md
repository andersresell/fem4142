# Clean-up plan

Progress tracker for the handover clean-up. Can be deleted when we're done.

## Done
- [x] Test suite in `tests/` (run with `python -m pytest`)
- [x] Bug fixes and removal of dead code
- [x] Rename `src/` to `fem/`
- [x] matplotlib 3.11 compatibility (colormap calls)

## Step 2: Code structure
- [x] Print a warning when the stiffness matrix is singular. Replace the Q4R/Q9R plate regression values (singular systems) with a test of the warning
- [x] IntEnum constants (`ElementType.Q16`, ...)
- [x] Dataclasses for Config, Mesh, SolverData, ...
- [x] One `ELEMENT_TRAITS` dict built once (including `il_to_ij`), replacing the parallel dicts in `fem_utils.py`
- [x] `morph_mesh` with named presets: `quarter_circle`, `quarter_annulus`, `quadrilateral`. Unknown name gives an error

## Step 3: Stress recovery
- [ ] Consistent and L2 (consistent mass) recovery as post-processing functions, not stored in solver_data
- [ ] Plane stress: σxx, σyy, σxy, von Mises
- [ ] Plate: Mxx, Myy, Mxy, Qx, Qy
- [ ] GUI: show the right quantities for each problem type, and a consistent/L2 switch

## Step 4: GUI
- [ ] `plot_field` function used by the GUI and usable from scripts
- [ ] Cache shape functions at the plot points
- [ ] Font and window sizes that work on different screens

## Step 5: Distribution
- [ ] Default: Python with numpy, scipy and matplotlib, clone the folder, scripts in the project root
- [ ] uv as fallback (`pyproject.toml`, `uv.lock`), which also allows `pip install -e .`
- [ ] README: tested versions, and run `python -m pytest` on a fresh install each semester

## Step 6: Documentation
- [ ] Update the README (new API, images)

## Later
- Where scripts live (project root for now)
- Selective reduced integration for plates
