#!/bin/bash

# python exhaustive_search.py

 PYTHONPATH=.   sinter collect \
     --metadata_func auto \
     --circuits circuits/HCultivationSurfaceCode/c=compiled-3q-multiplier*.stim \
     --decoders PymatchingGapSampler\
     --max_shots 1000_000_000_000 \
     --max_errors 1_00_000 \
     --custom_decoders "sampler:sinter_samplers" \
     --save_resume_filepath compiled_expansion.csv 

#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=expansion-H*,d1=2*.stim \
#      --decoders PymatchingGapSampler\
#      --max_shots 1000_000_000_000 \
#      --max_errors 1_00_000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath d2_expansion.csv 

# PYTHONPATH=.   sinter collect \
#     --metadata_func auto \
#     --circuits circuits/HCultivationSurfaceCode/c=expanded*atom*.stim \
#     --decoders PymatchingGapSampler\
#     --max_shots 1000_000_000_000 \
#     --max_errors 1_000_000 \
#     --custom_decoders "sampler:sinter_samplers" \
#     --save_resume_filepath atom_d3.csv 


# PYTHONPATH=.   sinter collect \
#     --metadata_func auto \
#     --circuits circuits/HCultivationSurfaceCode/c=end2end-*.stim \
#     --decoders PymatchingGapSampler\
#     --max_shots 1000_000_000_000 \
#     --max_errors 100_000_000 \
#     --custom_decoders "sampler:sinter_samplers" \
#     --save_resume_filepath end2end_$1.csv 

#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=expansion-*.stim \
#      --decoders PostSelectionSampler\
#      --max_shots 1000_000_000_000 \
#      --max_errors 1000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath stats_expansion_$1.csv 

#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=init*d1=5*.stim \
#      --decoders PostSelectionSampler\
#      --max_shots 100_000_000_000 \
#      --max_errors 1000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath stats_d5.csv 


#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=init*d1=3*.stim \
#      --decoders PostSelectionSampler\
#      --max_shots 1000_000_000 \
#      --max_errors 1000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath stats.csv 


#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=expanded*d1=3*.stim \
#      --decoders PymatchingGapSampler\
#      --max_shots 1_000_000_000 \
#      --max_errors 1_000_000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath expanded_stats.csv 

#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/CXCultivation/c=*.stim \
#      --decoders PostSelectionSampler\
#      --max_shots 100_000_000_000 \
#      --max_errors 1000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath stats.csv 

#  PYTHONPATH=.   sinter collect \
#      --metadata_func auto \
#      --circuits circuits/HCultivationSurfaceCode/c=init*d1=3*.stim \
#      --decoders PostSelectionSampler\
#      --max_shots 100_000_000_000 \
#      --max_errors 1000 \
#      --custom_decoders "sampler:sinter_samplers" \
#      --save_resume_filepath stats.csv 

