import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf

TIC_ID = 1461316
TFRECORD_DIR = "D:/ExoplanetMLFiles/Alt_Exoplanet/TESSExoplanetData/Astronet-Triage-master/Astronet-Triage-master/astronet/tfrecord"

def find_tce(tic_id, filenames):
  for filename in filenames:
    for record in tf.python_io.tf_record_iterator(filename):
      ex = tf.train.Example.FromString(record)
      if ex.features.feature["tic_id"].int64_list.value[0] == tic_id:
        print("Found {} in file {}".format(tic_id, filename))
        return ex
        raise ValueError("{} not found in files: {}".format(tic_id, filenames))

filenames = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "*"))
assert filenames, "No files found in {}".format(TFRECORD_DIR)
ex = find_tce(TIC_ID, filenames)

global_view = np.array(ex.features.feature["global_view"].float_list.value)
local_view = np.array(ex.features.feature["local_view"].float_list.value)
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].plot(global_view, ".")
axes[1].plot(local_view, ".")
plt.show()
