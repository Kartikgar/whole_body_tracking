MOTION_PATH = "data/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz"
import numpy as np

motion = np.load(MOTION_PATH)

for key in motion.keys():
    print(key)
    print("shape: ", motion[key].shape)
