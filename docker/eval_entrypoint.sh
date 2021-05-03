#!/bin/bash

#import dataset (lands in data folder in root) and unzip
gsutil -m cp -r gs://bucket-name/data ./
unzip data/"*.zip" -d data


# turn on bash's job control
set -m 

# start the primary process and put it in the background 
# start model script
scripts/evaluator.sh $@
