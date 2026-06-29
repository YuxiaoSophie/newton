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

* The user force has been tuned, and the isolated belt simulation is currently stable.

#### Future work

Could tune the belt parameters:

* mass
* stretch stiffness
* bend stiffness
* damping
* number of elements

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

Previous behavior issues:

* Some parts appeared on the floor because of coordinate / height mismatch.
* The belt dimension needed to match the real **248 mm × 168 mm × 6.6 mm** size.
* High stiffness caused instability: when user force pulled the belt, the rod tried to maintain stiff constraints and could explode.
* `add_rod_graph` builds the cable from an explicit graph topology: nodes, edges, and connection data must be provided manually. 
* `add_rod(..., closed=True)` builds the rod from an ordered list of points along one continuous path. For this belt, the geometry is just one closed ellipse, so `closed=True` automatically connects the last segment back to the first and is simpler and less error-prone.

Update:

The full `round_belt.py` scene is now more stable.

* Use the latest Newton source to reduce cable explosion

    This project now uses both:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install "newton[examples]"
    ```

    and the latest Newton source code:

    ```bash
    git clone https://github.com/newton-physics/newton
    ```

    The cloned Newton source is used because it may include newer solver fixes that are not yet available in the released `pip` package.

    The important improvement is in Newton’s VBD rigid contact behavior for finite-radius objects such as cables. Previously, when a small-radius cable contacted another object while rotating, the normal contact response could act at a rotating surface anchor point. This could inject artificial kinetic energy into the simulation, making the cable suddenly jump, spin, or explode. The newer Newton source improves this contact handling by applying the normal contact response more stably for cable-like objects, reducing non-physical energy gain during contact.

* Use center-of-mass body frames for rod segments

    The rod setup was also changed to use:

    ```python
    body_frame_origin="com"
    ```

    This places each rod segment’s body frame at its center of mass instead of at the segment start point.

Current video (Updated):

#### Next plan (Updated)

1. Add robot arms
2. Test more belt states:
   * dragging
   * lifting
   * twisting
   * folding
   * release and settling
