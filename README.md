# Round Belt Simulation Notes

This project develops a Newton/Warp round belt simulation.

## Expected folder structure

```text
my_project/
    round_belt.py
    round_belt.urdf.xacro
    task_board_urdf/
        common/
            table/
                table.obj
                table.mtl
            task_board_just_board.glb
        round_belt_task/
            round_belt_task_board/
                small_round_pulley/...
                large_round_pulley/...
        timing_belt_task/
            ...
```

## Run

**Terminal 1: Start the VirtualGL client**

```bash
/opt/VirtualGL/bin/vglclient
```

**Terminal 2: Run the simulation**

```bash
vglrun -d :1 python round_belt.py
```

## Files

### `test.py`

#### Current setup

This is the successful baseline example adapted from Newton’s `example_cable_twist.py`.

* Creates an elliptical closed-loop cable.
* Uses **64 elements**.
* Segment length is **0.1 m**, so the total cable length is about **6.4 m**.
* Cable radius is **0.02 m**, so the cable diameter is **0.04 m**.
* Full ellipse dimensions: approximately **2.56 m × 1.60 m**
* The rod uses Newton’s default mass/inertia setup.

#### Current status

* This example is stable.
* Its scale is much larger than the real belt, so the parameters cannot be directly reused.

---

### `only_belt.py`

#### Current setup

This is the isolated belt-only test. Goal is to match the real belt size: **248 mm × 168 mm × 6.6 mm**.
* Uses **48 elements**.
* Cable radius is **0.0033 m**, so the cable diameter is **0.0066 m**.
* Full ellipse dimensions: **0.248 m × 0.168 m**
* Uses softer belt parameters than `test.py`.
* Also tries softer user force / picking settings.
* Mass is not manually set in the script; this still needs to be tuned if necessary.

#### Current status

* Not fully stable yet.
* The belt can still become unstable when moved too much.

#### Next plan

Tune the belt parameters in this isolated file first:

* mass
* stretch stiffness
* bend stiffness
* damping
* number of elements
* user force / picking settings

---

### `round_belt.py`

#### Current setup

This is the full scene including:

* table
* board
* pulleys
* round belt on the table

The belt setup is based on the real belt dimensions:

* Target belt size: **248 mm × 168 mm × 6.6 mm**
* Uses **48 elements**.
* Cable radius is **0.0033 m**.
* Full ellipse dimensions: **0.248 m × 0.168 m**
* Mass is not manually set in the script; this still needs to be tuned if necessary.

#### Current status

The full scene is built, but the belt can still explode if moved too much.

Previous behavior issues:

* Some parts appeared on the floor because of coordinate / height mismatch.
* The belt dimension needed to match the real **248 mm × 168 mm × 6.6 mm** size.
* High stiffness caused instability: when user force pulled the belt, the rod tried to maintain stiff constraints and could explode.
* `add_rod_graph` builds the cable from an explicit graph topology: nodes, edges, and connection data must be provided manually. 
* `add_rod(..., closed=True)` builds the rod from an ordered list of points along one continuous path. For this belt, the geometry is just one closed ellipse, so `closed=True` automatically connects the last segment back to the first and is simpler and less error-prone.


#### Next plan

1. Tune the belt parameters in `only_belt.py`.
2. Copy the stable parameters into `round_belt.py`.
3. Test more belt states:
   * dragging
   * lifting
   * twisting
   * folding
   * release and settling
