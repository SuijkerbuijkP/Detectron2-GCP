import os

import detectron2.data.transforms as T
from adet.checkpoint import AdetCheckpointer
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from .loss_metrics import LossEvalHook
from .blendmask_mapper import BlendmaskMapperWithBasis


class COCOTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. As in "train_net", the build evaluator is implemented to do
    evaluation at specified points (TEST.EVAL_PERIOD) during training.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "training_eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        cfg = self.cfg.clone()
        cfg.defrost()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        """
        By default, in the default config settings (defaults.py), horizontal random flip resizeshortestedge are enabled. I would not change
        those variables through a custom loader, but just edit the respective settings in the cfg file.
        Specifically:
        Augmentations used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
        During testing flipping is also enabled.
        """
        # account for randomly rotated images
        # more augs can be added by using this strategy
        augs = [T.RandomRotation([-60.0, 60.0])]

        # just use if we run RCNN training
        if "RCNN" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


class AdetCOCOTrainer(COCOTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. As in "train_net", the build evaluator is implemented to do
    evaluation at specified points (TEST.EVAL_PERIOD) during training.
    """

    foldername = ""

    @staticmethod
    def set_foldername(cls, foldername):
        cls.foldername = foldername

    # Not sure if needed, but implemented in Adet train_net
    def resume_or_load(self, resume=True):
        if not isinstance(self.checkpointer, AdetCheckpointer):
            # support loading a few other backbones
            self.checkpointer = AdetCheckpointer(
                self.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        super().resume_or_load(resume=resume)

    @classmethod
    def build_train_loader(cls, cfg):
        # account for randomly rotated images
        # more augs can be added by using this strategy
        augs = [T.RandomRotation([-60.0, 60.0])]

        if "Blend" in cfg.MODEL.META_ARCHITECTURE:
            mapper = BlendmaskMapperWithBasis(cfg, is_train=True, augmentations=augs,
                                              foldername=cls.foldername)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    # for now validation loss during training does not work
    def build_hooks(self):
        hooks = DefaultTrainer.build_hooks(self)

        # use same augs as in build_train_loader
        # augs = [T.RandomRotation([-60.0, 60.0])]
        # hooks = super().build_hooks()
        # cfg = self.cfg.clone()
        #
        # cfg.defrost()
        # hooks.insert(-1, LossEvalHook(
        #     cfg.TEST.EVAL_PERIOD,
        #     self.model,
        #     build_detection_test_loader(
        #         self.cfg,
        #         self.cfg.DATASETS.TEST[0],
        #         BlendmaskMapperWithBasis(cfg, is_train=True, augmentations=augs,
        #                                  foldername=self.foldername)
        #     )
        # ))
        return hooks
