#!/bin/bash

mv train_entrypoint.sh entrypoint.sh

cd ..
sudo docker build -f  docker/Dockerfile -t eu.gcr.io/your-project-name/your-image-name ./
sudo docker push eu.gcr.io/your-project-name/your-image-name

read -p "Enter the dataset name:" dataset
read -p "Enter the run name:" run_name

# set arguments for specific run here as well, after the \
gcloud ai-platform jobs submit training $run_name --config gcloud_config.yaml --region 'europe-west4' --master-image-uri eu.gcr.io/your-project-name/your-image-name -- \ --run-name $run_name --dataset ./data/$dataset --area 500 --combine "scratch" "rust_main" 

cd docker
mv entrypoint.sh train_entrypoint.sh
