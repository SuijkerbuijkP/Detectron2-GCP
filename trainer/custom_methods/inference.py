import os

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

from custom_methods import load_checkpoint


def inference(cfg, args):
    """
    This function is used to perform inference. It loads the config file and with the corresponding eval_run
    parameter looks for the folder in which the to be evaluated model is saved. It loads a predictor, the latest
    model file and then performs inference. Pictures are saved to the "predictions" folder inside the corresponding
    run.
    """
    checkpoint_iteration, bucket = load_checkpoint(cfg, args)

    # set prediction threshold and model weights
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_" + checkpoint_iteration + ".pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.3
    cfg.freeze()

    # inference part
    predictor = DefaultPredictor(cfg)
    # save images in predictions folder, not required with pushing directly to GCP but cleaner
    if not os.path.isdir(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration)):
        os.mkdir(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration))

    # for d in DatasetCatalog.get("car_damage_test"):
    for d in DatasetCatalog.get("car_damage_val"):
        im = cv2.imread(d["file_name"])

        # save original image for easy comparison
        image_id = str(d["file_name"]).split("/")[-1].split(".")[0]
        image_extension = "." + str(d["file_name"]).split("/")[-1].split(".")[1]
        cv2.imwrite(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration) + "/" + image_id + image_extension, im)

        # produce predictions
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("car_damage_val"),
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )

        # .to("cpu"), which could be changed for large inference datasets, and save predictions
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration) + "/" + image_id + "_pred.jpeg",
                    v.get_image()[:, :, ::-1])

        # save to GCP
        blob = bucket.blob(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration)
                           + "/" + image_id + image_extension)
        blob.upload_from_filename(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration)
                                  + "/" + image_id + image_extension)
        blob = bucket.blob(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration)
                           + "/" + image_id + "_pred.jpeg")
        blob.upload_from_filename(cfg.OUTPUT_DIR + '/predictions' + str(checkpoint_iteration)
                                  + "/" + image_id + "_pred.jpeg")
