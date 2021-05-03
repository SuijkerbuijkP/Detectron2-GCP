import os

from google.cloud import storage
from google.cloud.storage import Bucket


def connect_to_bucket(bucket_name: str) -> Bucket:
    """
    @param bucket_name: name of the bucket of the save location
    @return: bucket object provided by GCP
    """
    storage_client = storage.Client(project='your-project-name')
    bucket = storage.Bucket(client=storage_client, name=bucket_name)

    return bucket


def get_available_folder(foldername: str, bucket_name: str, delimiter=None) -> str:
    """
    Looks for all the blobs in the bucket that begin with the prefix and returns the available folder.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you specify prefix ='a/', without a delimiter, you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a/' and delimiter='/', you'll get back:

        a/1.txt
    """

    bucket = connect_to_bucket(bucket_name)

    # define prefix (bucket folder name) which is used by notifier
    prefix = "model_output/" + foldername
    available_folder = None

    # this function checks if the last part of the name has a number, in case the input is some config file,
    # and adds _0 if not if there is already a folder with that name and the last part is not a number,
    # the same logic applies Note: Client.list_blobs requires at least package version 1.17.0.
    while not available_folder:
        # if this works, the folder exists
        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        blobs = list(blobs)
        if len(blobs) > 0:

            # split in parts and try to increment, if this fails add "_0" to the name
            m = foldername.split("_")
            try:
                m[-1] = int(m[-1]) + 1
                m[-1] = str(m[-1])
                foldername = "_".join(m)
                prefix = "model_output/" + foldername
            except:
                foldername = "_".join(m)
                foldername += "_0"
                prefix = "model_output/" + foldername
            continue
        else:
            # check if last part is a number, if not add _0
            try:
                int(prefix.split("_")[-1])
                available_folder = prefix
            except:
                available_folder = prefix + "_0"

    return available_folder


def load_checkpoint(cfg, args):
    """
    Loads specified checkpoint from specified bucket.
    """
    checkpoint_iteration = args.checkpoint
    bucket = connect_to_bucket(args.bucket)
    # load actual checkpoint
    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    blob = bucket.blob(cfg.OUTPUT_DIR + "/model_" + str(checkpoint_iteration) + ".pth")
    blob.download_to_filename(cfg.OUTPUT_DIR + "/model_" + str(checkpoint_iteration) + ".pth")
    if args.resume:
        # also write last checkpoint file for when --resume statement, model gets checkpoint name from this file
        with open(cfg.OUTPUT_DIR + "/last_checkpoint", "w") as file:
            file.write("model_" + str(checkpoint_iteration) + ".pth")
    # return statement not clean, but useful for inference code
    return checkpoint_iteration, bucket



