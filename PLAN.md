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
- [x] Mesh morphing, one function per shape: `morph_mesh_to_quarter_circle`, `morph_mesh_to_quarter_annulus`, `morph_mesh_to_quadrilateral`

## Step 3: Stress recovery
- [x] Consistent and L2 (consistent mass) recovery as post-processing functions, not stored in solver_data
- [x] Plane stress: σxx, σyy, σxy, von Mises
- [x] Plate: Mxx, Myy, Mxy, Vx, Vy
- [x] GUI: show the right quantities for each problem type, and a consistent/L2 switch

## Step 4: GUI
- [x] `plot_field` function used by the GUI and usable from scripts
- [x] Shape functions at the plot points calculated once per draw (first draw of 1800 Q4 elements: 6 s → 0.3 s)
- [x] Font sizes scale with the window height, and the window opens maximized
- [ ] Check the GUI by hand on a laptop and a large screen

## Step 5: Distribution
- [ ] Default: Python with numpy, scipy and matplotlib, clone the folder, scripts in the project root
- [ ] uv as fallback (`pyproject.toml`, `uv.lock`), which also allows `pip install -e .`
- [ ] README: tested versions, and run `python -m pytest` on a fresh install each semester

## Step 6: Documentation
- [ ] Update the README (new API, images)

## Later
- Where scripts live (project root for now)
- Selective reduced integration for plates
