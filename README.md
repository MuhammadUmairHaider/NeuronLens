# NeuronLens

> **Authors:** Muhammad Umair Haider, Hammad Rizwan, Hassan Sajjad, Peizhong Ju, A.B. Siddique
>
> **Link:** [TMLR / OpenReview](https://openreview.net/forum?id=AukyIhfBuW)
>
> **Abstract:** The conventional approach of attributing an entire neuron to a single semantic concept overlooks the inherent polysemanticity of neurons, leading to uncontrolled side effects during concept erasure. Through a systematic analysis of both encoder and decoder-based LLMs, this study demonstrates that while individual neurons encode multiple concepts, the activation magnitudes associated with different concepts follow distinct, Gaussian-like distributions with minimal overlap. Building upon this, we introduce **NeuronLens**, a novel framework that shifts the paradigm from discrete neuron-to-concept mapping to range-based attribution. By mapping specific activation ranges to individual concepts, NeuronLens enables precise, targeted interventions that maintain control while significantly reducing unintended interference with auxiliary concepts. Empirical validation proves NeuronLens outperforms traditional entire-neuron attribution methods in concept-erasure experiments.

This repository contains the official, self-contained codebase for our accepted paper. It focuses on exactly computing both targeted class and complement class accuracy, as well as tracking perplexity under different neuronal masking parameters.

## Core Features
The evaluation suite evaluates Llama 3.2 3B and GPT-2 models dynamically across intermediate activation layers to measure prediction consistencies, including the performance between class activation range masking and full neural masking frameworks.

**Supported Constraints:**
- Dynamically toggle evaluating LLama-3.2 3B or GPT-2 models.
- Native processing pipelines for Custom and Base Datasets (AG_News, Emotions, DBPedia-14).
- Parameterized execution for network scale variations: layer selection, neuron dropping limits, and subset capacities.

## Installation

Ensure you have Python 3.10+ and a CUDA-capable machine. 
To install the essential dependencies, activate your virtual environment and run the following pip command:

```bash
pip install -r requirements.txt
```

## Dataset Configuration

The codebase utilizes HuggingFace's datasets API to dynamically fetch models and base datasets for GPT-2:
- `fancyzhx/ag_news`
- `dair-ai/emotion`
- `dbpedia_14`

**GPT-2 Pre-requisite:**
Before evaluating GPT-2 across these datasets, the user must first fine-tune the model on the respective datasets to establish the baseline weights. The script expects the fine-tuned model weights to be present in the designated paths (e.g., `model_weights/fancyzhx/ag_news/weights.pth`).

**Llama Correct Predictions:**
For precise reproduction, Llama requires the pre-processed subset distributions tracking only accurate predictions to observe network stability metrics effectively. Unzip the corresponding .json sets into the `llama_correct_datasets/` folder structure:
- `correct_predictions_ag_news.json`
- `correct_predictions_emotions.json`
- `correct_predictions_DB_14.json`

## Evaluation Walkthrough

The central pipeline unifies everything into a single operational interface: `main.py`.

### Execution Example

We pass hyperparameters specifying model constraints to evaluate the base state, and subsequent masking states based on target activations.

```bash
python main.py --model llama --dataset db14 --layer 11 --percentage 0.3 --tao 2.0 --samples_per_class 400 --balance
```

**Available Hyperparameters:**
- `--model`: Options (`llama`, `gpt2`)
- `--dataset`: Options (`ag_news`, `emotions`, `db14`)
- `--layer`: The network layer representing the intermediate target to prune/constrain.
- `--percentage`: Cut-off factor indicating the percent of active network weight to sever for maximum mask isolation.
- `--tao`: Sigma coefficient bounds applied during parameter activation bounding (Range Mask). 
- `--samples_per_class`: Data constraint count applied equally per batch label structure. (e.g. `200`)
- `--balance`: Trims execution subset strictly normalized universally to the minimally prevalent class.

### Example Output
The output yields sequentially mapped states for precise analysis of stability deterioration natively tracking "Target" vs "Complement" relationships:
```text
=== Class X ===
Evaluating Base...
Applying Range Masking...
Applying Full Masking...

+---------+------------+---------+----------+----------+-----------+--------+
|  Class  |   State    | Tgt Acc | Tgt Conf | Comp Acc | Comp Conf |  PPL   |
+---------+------------+---------+----------+----------+-----------+--------+
| Class X |    Base    | ...     | ...      | ...      | ...       | 7.0074 |
|         | Range Mask | ...     | ...      | ...      | ...       | 7.4231 |
|         |  Max Mask  | ...     | ...      | ...      | ...       | 8.1250 |
+---------+------------+---------+----------+----------+-----------+--------+
```

## Citation
If this repository assists you in your research, please consider citing our work:
```bibtex
@inproceedings{insert_id,
  title={...},
  author={...},
  booktitle={...},
  year={2026}
}
```
