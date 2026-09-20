# CameraSetV1 physical-mount erratum

Status: `CameraSetV1.json` is retained unchanged as an immutable historical
artifact, but its `grasp_rgb` physical-role description and its resulting
`c0_eligible=true` conclusion are superseded by this erratum.

The physical mapping confirmed on site on 2026-09-20 is:

- `global_rgb`: GENERAL WEBCAM, legacy ACT key `handeye`; an external,
  stationary global camera.
- `grasp_rgb`: Sonix USB2.0_CAM1, legacy ACT key `fixed`; an eye-in-hand RGB
  camera rigidly mounted on the robot wrist and moving with the arm.
- `wrist_rgb`: Mech-Eye RGB stream; also wrist-mounted, but excluded from the
  current policy input because it failed the frozen throughput/latency gate.

The old stream device paths and static throughput/timestamp measurements remain
useful evidence. A new live preflight on 2026-09-20 confirmed the corrected
mapping at the current installation: `global_rgb` delivered 25.2 FPS with a
41.4 ms maximum observed frame age, and `grasp_rgb` delivered 25.1 FPS with a
14.5 ms maximum observed frame age. The preview showed the external global
view and the Sonix view between the gripper fingers, respectively.

Existing dynamic evidence is the completed run06 pick-and-place video at
`artifacts/run06_recovery/bamboo_newview_eval_run06_trial01/videos/observation.images.fixed/chunk-000/file-000.mp4`
(SHA-256 `8e4c8256d37dc372b076261535a5ef2d89b7bdaa1e7d15e0f7ab180180f468c2`).
It contains 1,491 frames at 640x480 and 25 FPS over 59.64 seconds, and shows the
same eye-in-hand view through approach, grasp, carry, and placement. This old
ACT video is camera-role and dynamic-visibility evidence only; it must not be
mixed into the C0 dataset.

Together, the current live preflight and existing motion video are sufficient
to accept the physical Sonix device as `grasp_rgb` without another robot-motion
trial solely for this correction. They do not make the incorrect metadata in
`CameraSetV1.json` true, and they do not authorize a formal C0 capture batch.
`CameraSetV2.json` is the new formal camera-set record with the corrected
mounting roles. `CameraSetV1.json` remains unchanged and its SHA-256 is not
reused by V2.
