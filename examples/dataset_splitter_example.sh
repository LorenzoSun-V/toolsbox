
data_dir=/data/nofar/person_behavior/labeled_data/labeling_rule_v2.0.0/20250709
config_file=configs/dataset_splitter.json

python preprocessing/dataset_splitter.py --source_dir ${data_dir} --config_file ${config_file}

# python preprocessing/dataset_splitter.py --config '{"fire-smoke": ["fire", "smoke"], "person": ["person"], "person_behavior": ["smoking", "tx"]}'