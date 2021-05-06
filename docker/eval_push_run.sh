#!/bin/bash
mv eval_entrypoint.sh entrypoint.sh

cd ..
sudo docker build -f  docker/Dockerfile -t eu.gcr.io/your-project-name/your-image-name ./
sudo docker push eu.gcr.io/your-project-name/your-image-name

# read parameters 
read -p "Enter the dataset name:" dataset_name
read -p "Enter the checkpoint:" checkpoint
read -p "Enter the eval-name:" eval_name
read -p "Enter the run name:" run_name

gcloud ai-platform jobs submit training $run_name --config eval_gcloud_config.yaml --region 'europe-west4' --master-image-uri eu.gcr.io/your-project-name/your-image-name -- \ --run-name $run_name --dataset ./data/$dataset_name --checkpoint $checkpoint --eval-name $eval_name --eval-only

cd docker
mv entrypoint.sh eval_entrypoint.sh
