MOTION_PATH = "logs/sim2sim_eval/2026.06.03/model_29999_motion_dataset_20260526_174909.npz"
import numpy as np

motion = np.load(MOTION_PATH)

for key in motion.keys():
    print(key)
    print("shape: ", motion[key].shape)
