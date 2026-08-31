# SO101 to AUBO i10 experimental workflows

This directory contains alternative SO101-leader teleoperation experiments. It is separate from the current
phone-to-AUBO pure-ACT workflow in `../phone_to_auboi10/`.

## Retained entry points

- `teleoperate.py`: direct SO101 joint-to-AUBO joint teleoperation using the main `AuboI10Robot` class.
- `calibrate.py`: collect SO101/AUBO correspondence points and write `calibration.json`.
- `teleop_fk.py`: calibrated forward-kinematics end-effector teleoperation.
- `teleop_ee_j6.py`: experimental end-effector position plus direct J6 control.

`calib_math.py`, `calibration.json`, and `so101_new_calib.urdf` support the calibrated FK workflows.
`aubo_sdk_reference/` contains retained AUBO SDK examples used as implementation references; the project does
not import them at runtime. `SO101/README.md` documents the SO101 model assets.

All entry points in this directory connect to physical hardware. Run them only after reviewing their control
mode, calibration, workspace bounds, and emergency-stop procedure; do not run multiple robot-control scripts
at the same time.
