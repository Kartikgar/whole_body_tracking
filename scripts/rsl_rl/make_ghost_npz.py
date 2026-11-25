#!/usr/bin/env python3
"""Convert a Genesis ``--record_motion`` NPZ into the plain ghost NPZ read by ``ghost_overlay.make_ghost_writer``.

Run in the Genesis env (the source pickles object arrays under numpy 2); the output has no object arrays, so the Isaac env
(numpy 1) reads it directly. Genesis frame 0 is the start state and frame t the state after t control steps.
"""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("genesis_npz")
p.add_argument("output_npz")
a = p.parse_args()
d = np.load(a.genesis_npz, allow_pickle=True)
pelvis = list(d["body_names"]).index("pelvis")
np.savez(a.output_npz, root_pos=d["body_pos_w"][:, :, pelvis].astype(np.float32),
         root_quat=d["body_quat_w"][:, :, pelvis].astype(np.float32), joint_pos=d["joint_pos"].astype(np.float32),
         joint_names=np.array([str(n) for n in d["joint_names"]]), valid_lengths=d["valid_lengths"].astype(np.int32))
print("wrote", a.output_npz, d["joint_pos"].shape)
