# Detectron2 Google Cloud Platform integration

## Introduction
Detectron2 is a computer vision library made by Facebook. Running this library on a local computer is well supported, but there is no guidance for running it on GCP's AI Platform, which requires a Docker image, and some further integration to be able to interact with the model output. The code in this repository has been used to train and evaluate models on GCP, and might be helpful for someone who wants to do something similar.

## Installation
AdelaiDet, Detectron2 and D2Go are added as submodules to this repository. To download the files after cloning do:

```sh
git submodule init
git submodule update
```

If you want to run the project locally, those packages have to be built as well. This can eb done by creating an environment with the requirements.txt textfile. If pytorch does not install correctly, install the tensorflow nightly through conda (assuming CUDA 11.1):

```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c conda-forge
```

Then, build D2, Adet and D2Go according to their respective instructions:

```sh
cd project/folder/here
python -m pip install -e detectron2
cd AdelaiDet
python setup.py build develop
cd ..
cd d2go
python -m pip install .
```

Building the Docker container should be done from root to include the trainer folder in the docker build context.
So, from root, call:

```sh
sudo docker build -f docker/Dockerfile -t my/image/name ./
```

# Running the framework
The model can be ran in a number of ways. It supports training from an Imagenet pretrained weightsfile by default, but can also be trained by loading a checkpoint from a previous model run.

A training job takes a valid training and validation .json annotation file from the datafolder in the vehicle-damaage bucket, and reads the images from the .zip files in this datafolder. Running a training job is done by doing:
```sh
cd docker
bash train_push_run.sh
```
The script builds and pushes to GCP, all you need to do is enter a valid job name. By default, training is performed with a weights file which is loaded from the Detectron2 or AdelaiDet model zoo. To load weights from a previous run, the trainer.sh file has to be edited. The parser takes a number of extra arguments:

- --local: if set, argument setting behaviour is changed slightly to better work with non-dockered runs
- --config-file: (valid!!) path to a different config file
- --run-name: takes the GCP run name, or a manually set name, which becomes the output folder
- --local: if ran locally, set this to get a prompt for an output folder name
- --bucket: name of the bucket that is worked in
- --architechture: important parameter, "adet", "d2go" and default Detectron2
- --dataset: part of the dataset path. Datasets are expected to have a name_train.json and name_val.json, with their images in an name_images folder. This structure is expected for all datasets. Setting a dataset is done as --dataset /path-to-folder/name_
- --num-classes: the number of classes present in the annotation file
- --resume: set to 'True', this will resume training of a certain run, from a certain checkpoint, with a certain number of iterations
- --reuse-weights: set to 'True', this will start a new training job but with the weights from a certain run, from a certain checkpoint
- --eval-only: takes no arguments but if present only performs inference on the validation dataset
- --eval-name: used to specify the run for resuming, reusing weights and inference. Should match the folder of the run to be used.
- --checkpoint: used to specify the checkpoint for resuming, reusing weights and inference
- --iterations: used to specify the extra number of iterations that training should resume when resume is specified
- --opts: optional arguments, mainly used to set MODEL.DEVICE cpu for local training
- extra arguments can be added to accommodate Hypertune hyperparameter training

It is also possible to filter jsons on runtime. This is done through the "preprocess.py" file and can be enabled by passing --filter to the parser. This gives some more settings:

- --input: path to an input json. This is set automatically when not running locally
- --output: path to the output json. This is set automatically when not running locally
- --categories: list of names, seperated with a space (e.g. dogs chairs)
- --area: integer giving the max area of annotations. Smaller annotations are filtered out
- --merge: integer giving the max area of annotations. Smaller annotations will be merged with close annotations of the same category
- --combine: list of names, seperated with a space, that are combined into a single category
- --y: skips the overwrite warning. Set automatically when not running locally.

Inference is split from the training job already, with different scripts. This way, you do not need to change parameters every time you want to do something different. They are ran similarly to a training, but with the corresponding .sh file.

# Results
The results of all job types are saved to and loaded from the GCP bucket. Make sure the correct run and checkpoint is entered when loading from the bucket. By default, the output folder is equal to the config used, incremented by 1 for each new run. Test metrics can be read in
tensorboard, which can be called from the bucket directly if you are blessed enough to not have a Windows machine and have Tensorflow installed. It can be called with:

```sh
tensorboard --host 0.0.0.0 --logdir gs://vehicle-damage/model_output
```
where the host is 0.0.0.0 optional to allow connections with a VM if it is not ran locally.
