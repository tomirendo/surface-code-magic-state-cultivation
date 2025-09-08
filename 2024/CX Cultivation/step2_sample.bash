#!/usr/bin/bash

 PYTHONPATH=.   sinter collect \
     --metadata_func auto \
     --circuits erasure_circuits/c=no-erasure*.stim \
     --decoders SingleObservableGapSampler \
     --max_shots 1_000_000_000 \
     --max_errors 10_000 \
     --custom_decoders "sampler:sinter_samplers" \
     --save_resume_filepath stats.csv
     
 PYTHONPATH=.  sinter collect \
     --metadata_func auto \
     --circuits circuits/c=init*d1=3*.stim \
    --decoders PostSelectionSampler\
     --max_shots 40_000_000_000 \
     --max_errors 1000 \
     --save_resume_filepath stats.csv \
     --custom_decoders "sampler:sinter_samplers" \


 PYTHONPATH=.   sinter collect \
     --metadata_func auto \
     --circuits circuits/c=end2end*d2=11*p=0.001*.stim \
     --decoders GapSampler \
     --max_shots 1_000_000_000_000 \
     --max_errors 100_000_000 \
     --custom_decoders "sampler:sinter_samplers" \
     --save_resume_filepath stats.csv

#PYTHONPATH=.   sinter collect \
#    --metadata_func auto \
#    --circuits circuits/c=init*d1=5*.stim \
#    --decoders PostSelectionSampler\
#    --max_shots 100_000_000_000 \
#    --max_errors 1000 \
#    --save_resume_filepath stats.csv \
#    --custom_decoders "sampler:sinter_samplers" 
#

