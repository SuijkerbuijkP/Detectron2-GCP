#!/usr/bin/env python
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation and supports models from Detectron2,
AdelaiDet and D2Go. The argument parser is made in such a way that runs can be set up in a flexible way
on runtime, supporting merging, filtering and dataset selection.
"""

import os
import subprocess
import traceback

from d2go.runner import GeneralizedRCNNRunner
from d2go.setup import setup_after_launch
from detectron2.config import get_cfg
from adet.config import get_cfg as adet_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_setup, PeriodicWriter, launch

from custom_methods import inference, load_checkpoint, get_parser, get_available_folder
from custom_trainers import COCOTrainer, LossMetricWriter, AdetCOCOTrainer
from data import preprocess


def setup(args):
    """
    Create configs and perform basic setups.
    """
    if args.architecture.lower() == "d2go":
        trainer = GeneralizedRCNNRunner()
        # as cfg is grabbed from trainer in this architecture
        cfg = trainer.get_default_cfg()
    # adet should (as far as I've tested) also just run mask-rcnn, but I keep it separate just in case
    elif args.architecture.lower() == "adet":
        cfg = adet_cfg()
        trainer = None
    else:
        cfg = get_cfg()
        trainer = None
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # ADD HYPERTUNE ARGUMENTS HERE FOR HYPERPARAMETER TUNING (for example cfg.SOLVER.learning_rate = args.lr)
    cfg.SOLVER.IMS_PER_BATCH = int(args.batchsize)

    # set eval period for validation during training (epochs = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES)
    cfg.TEST.EVAL_PERIOD = 500  # easy for now

    # set number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(args.num_classes)

    # Ask for run name if ran locally
    if args.local:
        args.run_name = input("Give output folder a name: ")

    # if a single gpu is used this needs to be set, error otherwise
    if args.num_gpus == 1 and args.architecture.lower() == "adet":
        cfg.MODEL.BASIS_MODULE.NORM = "BN"

    # checks what type of run is required and loads/sets variables accordingly
    if args.eval_only:
        if args.checkpoint == "" or args.eval_name == "":
            raise Exception("No checkpoint or previous run provided, please set with --eval-name and --checkpoint.")
        cfg.OUTPUT_DIR = "model_output/" + args.eval_name
    elif args.resume:
        if args.checkpoint == "":
            raise Exception("No checkpoint provided, please set with --eval-name and --checkpoint.")
        # loads checkpoint from which we continue, to be set with args.checkpoint
        cfg.OUTPUT_DIR = "model_output/" + args.eval_name
        load_checkpoint(cfg, args)
    elif args.reuse_weights == "True":
        if args.checkpoint == "":
            raise Exception("No checkpoint provided, please set with --eval-name and --checkpoint.")
        # load model weights from defined run
        cfg.OUTPUT_DIR = "model_output/" + args.eval_name
        checkpoint_iteration, bucket = load_checkpoint(cfg, args)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_" + checkpoint_iteration + ".pth")
        # if hyperparameter tuning is done, the name does need to be checked and set to new output folder
        cfg.OUTPUT_DIR = get_available_folder(args.run_name, args.bucket)
    else:
        # if hyperparameter tuning is done, the name does need to be checked and set to new output folder
        cfg.OUTPUT_DIR = get_available_folder(args.run_name, args.bucket)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg, trainer


def main(args):
    try:
        cfg, trainer = setup(args)

        # filter and make .npz files needed for Blendmask
        if args.filter or args.architecture == "adet":
            if not args.local:
                # no need to set this manually, as we can overwrite json in container
                args.input = args.dataset + "train.json"
                args.output = args.dataset + "train.json"
                args.y = True  # set this just in case it is missed
            preprocess(args)

        if not args.filter or args.architecture != "adet":
            # register dataset so that it can be used train and val images can live in the same folder, as the image
            # id's are unique so only need to define the correct .json
            register_coco_instances("car_damage_train", {}, args.dataset + "train.json", args.dataset + "images")
            register_coco_instances("car_damage_val", {}, args.dataset + "val.json", args.dataset + "images")

        # overwrite trainer if not d2go
        if args.architecture.lower() == "adet" and args.architecture.lower() != "d2go":
            AdetCOCOTrainer.foldername = args.dataset.split('/')[-1] + "images"
            trainer = AdetCOCOTrainer(cfg)
        elif args.architecture.lower() != "d2go":
            trainer = COCOTrainer(cfg)

        # TODO: add evaluation to this
        if args.architecture.lower() == "d2go":
            setup_after_launch(cfg, cfg.OUTPUT_DIR, trainer)  # some d2go magic required, crash if removed
            model = trainer.build_model(cfg)
            return trainer.do_train(cfg, model, resume=args.resume)

        # eval logic for adet and d2
        if args.eval_only:
            return inference(cfg, args)

        # include the hook that reports to CloudML Hypertune
        writers = [LossMetricWriter()]
        trainer.register_hooks(
            [PeriodicWriter(writers)]
        )

        trainer.resume_or_load(resume=args.resume)
        if args.resume:
            # resume takes old settings, so add iterations that we want to run
            cfg.defrost()
            cfg.SOLVER.MAX_ITER += int(args.iterations)
        return trainer.train()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        # kill map watcher at the end of model execution
        bash_command = "pkill inotifywait"
        subprocess.run(bash_command, shell=True)


if __name__ == "__main__":
    args = get_parser().parse_args()

    # start training, if eval only do not use launch because it works properly, but errors out in the end (annoying)
    if args.eval_only:
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            dist_url="auto",
            args=(args,),
        )
