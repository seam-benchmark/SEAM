import os.path
import shutil
from argparse import ArgumentParser
import json
from model_wrappers.hf_pipline_wrap import HfPipelineWrap
import gc
import torch
from tqdm import tqdm
import pickle as pkl

QUANTIZED_MODELS = {'mistralai/Mixtral-8x22B-Instruct-v0.1': '4bit'}


def read_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    fill_default_arguments(meta)
    return meta


def handle_out_path_overrides(metadata):
    output_dir = os.path.join(metadata['out_dir'], metadata['run_name'])
    out_dir_exists = os.path.exists(output_dir)
    override_all = metadata['override']
    path_to_metadata = os.path.join(output_dir, "metadata.json")
    run_mode = 'resume'

    if out_dir_exists and not override_all:
        print("Warning: resuming run from current point")
    if override_all:
        run_mode = 'override'
        if out_dir_exists:
            shutil.rmtree(output_dir)
            print("Warning: out_dir exists, overriding all files in it.")
        else:
            print("Warning: out_dir does not exist, has nothing to override. Ignoring override flag.")

    os.makedirs(output_dir, exist_ok=True)
    with open(path_to_metadata, 'w') as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Saving the current metadata to {path_to_metadata}")

    return output_dir, run_mode


def fill_default_arguments(metadata):
    metadata['resume'] = metadata.get('resume', False)
    metadata['temperature'] = metadata.get('temperature', 0.8)
    metadata['batch_size'] = metadata.get('batch_size', 8)
    metadata['num_demonstrations'] = metadata.get('num_demonstrations', 3)


def run_all(all_datasets, metadata, output_dir, truncation, run_mode, model_name):
    model = load(model_name, metadata)
    for dataset in all_datasets:
        all_ds_instances = dataset["instances"]
        print(f"\nRunning model {model_name} on dataset {dataset['name']}")
        for i, ds_instance in tqdm(enumerate(all_ds_instances), total=len(all_ds_instances)):
            out = os.path.join(output_dir, dataset['name'], f"{model_name}_{i}.json")
            if os.path.exists(out) and run_mode == 'resume':
                print(f"Output file {out} exists, skipping...")
                continue
            if truncation == 'max':
                truncation = model.get_max_window()
            ds_instance.predict(model=model,
                                out_path=out,
                                num_truncation_tokens=truncation)
            gc.collect()


def load(model_name, metadata):
    gc.collect()
    torch.cuda.empty_cache()
    load_in_4_bit = load_in_8_bit = False
    if model_name in QUANTIZED_MODELS:
        if QUANTIZED_MODELS[model_name] == '4bit':
            load_in_4_bit = True
        elif QUANTIZED_MODELS[model_name] == '8bit':
            load_in_8_bit = True
        else:
            print("QUANTIZATION MODE NOT SUPPORTED. LOADING IN DEFAULT DTYPE.")

    model = HfPipelineWrap(model_name, metadata['temperature'], metadata['batch_size'],
                           load_in_4_bit=load_in_4_bit, load_in_8_bit=load_in_8_bit)
    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_truncation_strategy(metadata):
    truncation = metadata.get('truncation_strategy')  # one of {max, min, set}. If set, max_num_tokens must be set.
    max_num_tokens = metadata.get('max_num_tokens')
    if truncation is None:
        raise ValueError("truncation_strategy must be set.")
    if truncation == 'max' and max_num_tokens is not None:
        raise ValueError("If truncation_strategy is 'max', max_num_tokens must be None.")
    elif truncation == 'set':
        if max_num_tokens is None:
            raise ValueError("If truncation_strategy is 'set', max_num_tokens must be set.")
        truncation = max_num_tokens

    return truncation


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()
    metadata_dict = read_metadata(args.config)
    out_dir, running_mode = handle_out_path_overrides(metadata_dict)
    datasets_pickle = os.path.join(out_dir, metadata_dict["datasets_pickle_path"])
    print(f"Loading datasets from {datasets_pickle}")
    with open(datasets_pickle, 'rb') as f:
        datasets_list = pkl.load(f)
    truncation_strategy = get_truncation_strategy(metadata_dict)
    print(f"run output dir in {out_dir}")
    run_all(all_datasets=datasets_list, metadata=metadata_dict, output_dir=out_dir,
            truncation=truncation_strategy, run_mode=running_mode, model_name=args.model_name)

