# SEAM: A Stochastic Benchmark for Multi-Document Tasks

Project website: [https://seam-benchmark.github.io](https://seam-benchmark.github.io).

## Download the data
Some of the datasets included in SEAM can be reached via huggingface datasets (FuseReviews, MultiNews, Scico). 

For the rest of the datasets, please download from [here](https://drive.google.com/file/d/1H6pBzwJmCfFGOWOOzyLDFID2lIk9bbfI/view?usp=share_link) the zip containing the data and unzip it before generating a benchmark instance.

The path where you unzip the data should appear in the configuration file as examplified in the `files/configuration/generate.json` file (it should be filled within the `path` field for each dataset among MusiQue, ECB, and OpenASP).

## Prepare the environment

To set up the running environment, run the following command:
```
git clone git@github.com:seam-benchmark/SEAM.git
cd SEAM
export PYTHONPATH=./
python3.11 -m venv <PATH_TO_VENV>
source <PATH_TO_VENV>/bin/activate
pip install -r requirements.txt
```

## Generate a benchmark instance

SEAM is a *benchmark generator* for multi-document tasks. 
To generate such instances, you need to provide a configuration file that specifies 
the datasets to include in the benchmark instance, how many few-shot examples to include in a prompt, 
the number of resampling to perform, and how many instances to sample from each dataset. 
For reproducibility, you can also specify a random seed.

An example for such a configuration file is provided in [`files/configuration/generate.json`](files/configuration/generate.json).

To generate a benchmark instance, run the following command:
```
python scripts/generate_benchmark.py --config files/configuration/generate.json
```

## Run predictions
After generating a benchmark instance, you can run predictions over the generated datasets 
using the configuration file [`files/configuration/predict.json`](files/configuration/predict.json).

The `predict.json` file contains the path to the generated benchmark from previous step, the batch size, and the decoding temperature for the LLMs.

To run predictions, run the following command:
```
bash scripts/run_all.sh --config files/configuration/predict.json
```

To control the models to evaluate SEAM on, you can modify the [`scripts/run_all.sh`](scripts/run_all.sh) file.

## Evaluate the predictions

To evaluate the predictions, you can use the script [`scripts/eval_all.sh`](scripts/eval_all.sh) by providing 
the  path to the predictions from previous step, and output path where all results will be saved.

```
bash scripts/eval_all.sh --predictions_path <PATH_TO_PREDICTIONS> --output_path <OUTPUT_PATH>
```

## Reproduce the results from the paper

The example jsons in the `files/configuration` directory are set to reproduce the results from the paper, 
as well as `run_all.sh` and `eval_all.sh` scripts.

LLMs predictions over the benchmark instance as presented in the paper, can be found [here](https://drive.google.com/drive/folders/1d8sJIwaL-sEhycWrnwSXIT_3FnOXmxSY?usp=sharing)
