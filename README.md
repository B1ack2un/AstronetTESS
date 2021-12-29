# Astronet Neural Networks for Windows: A Neural Network for Analyzing TESS Data
![Transit Animation](docs/transit.gif)

## Background

This directory contains TensorFlow models and data processing code for identifying exoplanets in astrophysical light curves. 
This model is derived from Liang Yu's code. For a complete background on how this model works, see [Shallue & Vanderburg's paper](http://adsabs.harvard.edu/abs/2018AJ....155...94S) in
as well as [Liang Yu's paper](https://ui.adsabs.harvard.edu/abs/2019AJ....158...25Y/abstract) in *The Astronomical Journal*.
Both the triage and vetting networks are included in this project.

For shorter summaries, see:

* ["Earth to Exoplanet"](https://www.blog.google/topics/machine-learning/hunting-planets-machine-learning/) on the Google blog
* [This blog post](https://www.cfa.harvard.edu/~avanderb/page1.html#kepler90) by Andrew Vanderburg
* [This great article](https://milesobrien.com/artificial-intelligence-gains-intuition-hunting-exoplanets/) by Fedor Kossakovski
* [NASA's press release](https://www.nasa.gov/press-release/artificial-intelligence-nasa-data-used-to-discover-eighth-planet-circling-distant-star) article

## Citation
Original papers from which this project is largely derived:

Yu, L. et al. (2019). Identifying Exoplanets with Deep Learning III: Automated Triage and Vetting of TESS Candidates. *The Astronomical Journal*, 158(1), 25.

See also the original Shallue & Vanderburg paper:

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep
Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet
around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Full text available at [*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

## Code Directories

[astronet/](astronet/)

* [TensorFlow](https://www.tensorflow.org/) code for:
  * Downloading and preprocessing TESS data.
  * Building different types of neural network classification models.
  * Training and evaluating a new model.
  * Using a trained model to generate new predictions.

[light_curve_util/](light_curve_util)

* Utilities for operating on light curves. These include:
  * Reading TESS data from `.fits` files.
  * Phase folding, splitting, binning, etc.
* In addition, some C++ implementations of light curve utilities are located in
[light_curve_util/cc/](light_curve_util/cc).

[third_party/](third_party/)

* Utilities derived from third party code.

## Walkthrough

### Install Required Packages

First, ensure that you have installed the following required packages:

* **Anaconda Navigator** ([instructions](https://docs.anaconda.com/anaconda/install/windows/))
* **PIP** ([instructions](https://pip.pypa.io/en/stable/installation/))
* **Python Version 3.7** ([instructions](https://stackoverflow.com/questions/61491893/i-cannot-install-tensorflow-version-1-15-through-pip))
* **TensorFlow** ([instructions](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/))
  - **TensorFlow V1.15** ([instructions for downgrading](https://www.tensorflow.org/install/pip))
* **Pandas** ([instructions](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html))
* **NumPy** ([instructions](https://docs.scipy.org/doc/numpy/user/install.html))
* **AstroPy** ([instructions](http://www.astropy.org/))
* **PyDl** ([instructions](https://pypi.python.org/pypi/pydl))
* **Bazel** ([instructions](https://docs.bazel.build/versions/main/install-windows.html))
* **Windows Subsystem for Linux** ([instructions](https://docs.microsoft.com/en-us/windows/wsl/install))

### Bazel Installation

Since Bazel can be a bit difficult to install, a brief section is devoted to the installation process for Bazel. 
If there are still difficulties with installing this program, then it is still possible to run the original python scripts but it is possible that the user will encounter 
specific issues with recognition of FLAGS in running the generate_input_records python file which creates the necessary TFRecords files for analysis. 

The first two steps of the installation are relatively straightforward and will not be covered here. Starting from step 3, the version of Bazel to install would be 
the latest release on GitHub. At the time of writing, this would be Bazel 4.2.2. Next, search for the file named "bazel-4.2.2-windows-x86_64.exe" and click it. A download
should begin shortly. Once completed, this binary file should be moved and copied to the Astronet-Triage-master and Astronet-Vetting-master folders. Additionally, the binary 
file should be renamed to "bazel.exe". Following step 4, access to this binary file should be added through the PATH variable in the following format in the command prompt:
<p align="center">
    set PATH = %PATH%; < path to the Bazel binary >
</p>

 However, this only works temporarily so in order to permanently set the PATH variable use this instead:
 <p align="center">
    setx PATH "i.e. path to bazel binary;%PATH%"
</p>
<p align="center">
    set PATH = i.e. path to Bazel binary;%PATH%	
</p>

Before testing Bazel, a couple more things need to be taken care of prior to confirmation as shown in Step 5. 
First, [MSYS2](/https://www.msys2.org/) needs to be installed on the machine. Next, the BAZEL_SH variable needs to be defined in the environment variables. 
This can be found [here](https://docs.bazel.build/versions/main/windows.html).
After this, step 5 can be followed for verification. 

### Installation Troubleshooting

If there are more issues with installing Bazel, check [here](https://github.com/bazelbuild/bazel/issues) to see if your specific problem has been identified 
yet.

### Testing Programs
Prior to analyzing this code, start with opening the Astronet-Triage-master folder in the Command Prompt and running Bazel by testing the code in the various subfolders with
the following command:
<p align="center">
    bazel test astronet/... light_curve_util/... third_party/... --test_arg=--test_srcdir=/path_to_Astronet-Triage-master_directory/Astronet-Triage-master/Astronet-Triage-master/
</p>

It is best to check each of the major folders individually so that any errors which may arise in testing can be easily identified and corrected. 
Once all of these folders have been verified to be correct, then the next step is to download the TESS files. 

### Downloading TESS Data

A *light curve* is a plot of the brightness of a star over time. We will befocusing on light curves produced by the TESS space telescope. 
An example light curve (produced by Kepler) is shown below.

![Kepler-934](docs/kepler-943.png)

To train a model to identify planets in TESS light curves, you will need atraining set of labeled *Threshold Crossing Events* (TCEs). 
A TCE is a periodic signal that has been detected in a light curve, and is associated with a *period* (the number of days between each occurrence of the detected signal),
a *duration* (the time taken by each occurrence of the signal), an *epoch* (the time of the first observed occurrence of the signal), and possibly additional
metadata like the signal-to-noise ratio. An example TCE is shown below. The labels are ground truth classifications (decided by humans) that indicate which
TCEs in the training set are actual planets signals and which are caused by other phenomena.

![Kepler-934 Transits](docs/kepler-943-transits.png)

Since the data which will form the basis for the training set will be selected by the user, the relevant TESS data must be downloaded from the [Mikulski Archive for Space 
Telescopes](https://archive.stsci.edu/missions-and-data/tess). This site contains all of the data relevant to the TESS mission. In order to download the .csv file for all of the 
recorded TCEs in the desired sector, go [here](https://archive.stsci.edu/tess/bulk_downloads.html) and click on the TCE Bulk Downloads link. From here, pick the desired sector 
and download the .csv file by clicking on the link next to the sector number.    

Next, the TIC IDs of the TCEs must be copied and pasted into a separate .txt file. This .txt file will be named sector-x-yyy.txt where x identifies the sector number of the TCE 
file. 

After saving the .txt file, the executable file must be created by navigating to the astronet/data folder and adjusting the BUILD file. This BUILD file is directly responsible 
for the creation of the executable files which will be run in WSL. For more information on how the BUILD files work check [here](). Add the following to the BUILD file 
running Bazel with the following command:
<p align="center">
    bazel build data/...
</p>
