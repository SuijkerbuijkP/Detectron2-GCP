import argparse


def get_parser():
    #######################################################################################################
    # MODEL ARGUMENTS
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="",  # has to be a valid path, model checks before use
                        metavar="FILE", help="path to config file",
                        )
    parser.add_argument("--run-name",
                        help="Name of this run which is set as output folder. Set through ai platform job name.",
                        )
    parser.add_argument("--local", action="store_true", help="If local, ask for run-name in terminal.", )
    parser.add_argument("--bucket", default="vehicle-damage", help="Specify the bucket to work with")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    # goal specific arguments, like training with previous weights, normal training or inference
    parser.add_argument("--architecture", default="",
                        help="If you want to use AdelaiDet to for example use Blendmask, specify this here",
                        )
    parser.add_argument("--dataset", default="./data/",
                        help="""Enter name of the to be processed dataset, with trailing _. Default is './data/', 
                        example is './data/synth_'.""",
                        )
    parser.add_argument("--num-classes", default=7, help="Number of classes in the dataset", )
    parser.add_argument("--resume", help="""
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """, default=False,
                        )
    parser.add_argument("--reuse-weights", default=False,
                        help="Starts a new run with weights from checkpoint defined by --eval-run and --checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--checkpoint", help="Specify the iteration number or *final*.", default="")
    parser.add_argument("--eval-name",
                        help="This should be the same name of the run that is being evaluated. Sets output folder, "
                             "just as --run-name, but as this is set through GCP and has to be unique, a second "
                             "parameter for eval is required.",
                        default=""
                        )
    parser.add_argument("--iterations", help="Specify the additional number of iterations if --resume.", default="")
    #######################################################################################################
    # ADD HYPERTUNE ARGUMENTS HERE FOR HYPERPARAMETER TUNING (for example --lr)
    parser.add_argument("--batchsize", default=2)
    #######################################################################################################
    # FILTER ARGUMENTS
    parser.add_argument("--filter", action='store_true', help="Set to true if filtering is required")
    parser.add_argument("-y", "--y", action='store_true', help="If this is set, skip asking for overwrite")
    parser.add_argument("-i", "--input", help="path to a json file in coco format")
    parser.add_argument("-o", "--output", help="path to save the output json")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories",
                        help="List of filter category names separated by spaces, e.g. -c person dog bicycle")
    parser.add_argument("-a", "--area", type=int, help="Area that should be filtered out, e.g. -a 900")
    parser.add_argument("-m", "--merge", type=int, help="Area that should be merged, e.g. -m 900")
    parser.add_argument("--combine", nargs='+', help="Categories that should be combined, e.g. rust_sub rust_main")
    #######################################################################################################
    # optional arguments, like MODEL.DEVICE cpu
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[],
                        nargs=argparse.REMAINDER, )
    return parser
