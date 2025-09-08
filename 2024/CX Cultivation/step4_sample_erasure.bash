#!/usr/bin/bash


for i in {1..1000}; do
    python generate_erasure.py

 PYTHONPATH=.   sinter collect \
     --metadata_func auto \
     --circuits erasure_circuits/c=erasure*.stim \
     --decoders SingleObservableGapSampler \
     --max_shots 10_000 \
     --max_errors 10_000 \
     --custom_decoders "sampler:sinter_samplers" \
     --save_resume_filepath erasure_stats/erasure_stats_$i.csv

done
