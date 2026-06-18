# Import necessary modules
import copy
import json
import os
import time
import torch
import torch.nn as nn
import pyarrow.parquet as pq

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch


def _get_local_zeroshot_data_root():
    return os.environ.get("LOCAL_ZEROSHOT_DATA_ROOT", "/2T/zhuhe/data/zero-shot")


def _read_parquet_rows(path):
    return pq.read_table(path).to_pylist()


def _read_jsonl_rows(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_local_boolq_dataset():
    base_dir = os.path.join(_get_local_zeroshot_data_root(), "boolq", "data")
    train_path = os.path.join(base_dir, "train-00000-of-00001.parquet")
    validation_path = os.path.join(base_dir, "validation-00000-of-00001.parquet")
    if not (os.path.exists(train_path) and os.path.exists(validation_path)):
        return None

    def normalize(rows):
        normalized = []
        for row in rows:
            item = dict(row)
            if "label" not in item and "answer" in item:
                item["label"] = bool(item["answer"])
            normalized.append(item)
        return normalized

    return {
        "train": normalize(_read_parquet_rows(train_path)),
        "validation": normalize(_read_parquet_rows(validation_path)),
    }


def _load_local_hellaswag_dataset():
    base_dir = os.path.join(_get_local_zeroshot_data_root(), "hellaswag", "data")
    train_path = os.path.join(base_dir, "train-00000-of-00001.parquet")
    validation_path = os.path.join(base_dir, "validation-00000-of-00001.parquet")
    test_path = os.path.join(base_dir, "test-00000-of-00001.parquet")
    if not (os.path.exists(train_path) and os.path.exists(validation_path)):
        return None
    dataset = {
        "train": _read_parquet_rows(train_path),
        "validation": _read_parquet_rows(validation_path),
    }
    if os.path.exists(test_path):
        dataset["test"] = _read_parquet_rows(test_path)
    return dataset


def _normalize_arc_row(row):
    item = dict(row)
    question = item.get("question")
    if isinstance(question, dict):
        item["question"] = question.get("stem", "")
        choices = question.get("choices", [])
        item["choices"] = {
            "label": [choice.get("label", "") for choice in choices],
            "text": [choice.get("text", "") for choice in choices],
        }
    elif isinstance(item.get("choices"), list):
        choices = item["choices"]
        item["choices"] = {
            "label": [choice.get("label", "") for choice in choices],
            "text": [choice.get("text", "") for choice in choices],
        }
    return item


def _find_local_arc_challenge_files():
    root = _get_local_zeroshot_data_root()
    parent = os.path.dirname(root)
    candidates = [
        os.environ.get("LOCAL_ARC_CHALLENGE_DIR"),
        os.path.join(root, "arc_challenge"),
        os.path.join(root, "ARC-Challenge"),
        os.path.join(root, "ARC-V1-Feb2018", "ARC-Challenge"),
        os.path.join(root, "ARC-V1-Feb2018-2", "ARC-Challenge"),
        os.path.join(parent, "ARC-V1-Feb2018", "ARC-Challenge"),
        os.path.join(parent, "ARC-V1-Feb2018-2", "ARC-Challenge"),
    ]
    for base_dir in candidates:
        if not base_dir:
            continue
        train_path = os.path.join(base_dir, "ARC-Challenge-Train.jsonl")
        validation_path = os.path.join(base_dir, "ARC-Challenge-Dev.jsonl")
        test_path = os.path.join(base_dir, "ARC-Challenge-Test.jsonl")
        if os.path.exists(validation_path):
            return {
                "train": train_path,
                "validation": validation_path,
                "test": test_path,
            }
    return None


def _load_local_arc_challenge_dataset():
    files = _find_local_arc_challenge_files()
    if files is None:
        return None
    validation_rows = [_normalize_arc_row(row) for row in _read_jsonl_rows(files["validation"])]
    dataset = {
        "train": [_normalize_arc_row(row) for row in _read_jsonl_rows(files["train"])]
        if os.path.exists(files["train"])
        else list(validation_rows),
        "validation": validation_rows,
    }
    if os.path.exists(files["test"]):
        dataset["test"] = [_normalize_arc_row(row) for row in _read_jsonl_rows(files["test"])]
    return dataset


def _resolve_eval_tasks(task_names):
    from lm_eval.tasks.arc import ARCChallenge as HarnessARCChallenge
    from lm_eval.tasks.hellaswag import HellaSwag as HarnessHellaSwag
    from lm_eval.tasks.superglue import BoolQ as HarnessBoolQ

    local_datasets = {
        "boolq": _load_local_boolq_dataset(),
        "hellaswag": _load_local_hellaswag_dataset(),
        "arc_challenge": _load_local_arc_challenge_dataset(),
    }

    class LocalBoolQ(HarnessBoolQ):
        EVAL_HARNESS_NAME = "boolq"

        def download(self, data_dir=None, cache_dir=None, download_mode=None):
            dataset = local_datasets["boolq"]
            if dataset is None:
                return super().download(data_dir, cache_dir, download_mode)
            self.dataset = copy.deepcopy(dataset)

    class LocalHellaSwag(HarnessHellaSwag):
        EVAL_HARNESS_NAME = "hellaswag"

        def download(self, data_dir=None, cache_dir=None, download_mode=None):
            dataset = local_datasets["hellaswag"]
            if dataset is None:
                return super().download(data_dir, cache_dir, download_mode)
            self.dataset = copy.deepcopy(dataset)

    class LocalARCChallenge(HarnessARCChallenge):
        EVAL_HARNESS_NAME = "arc_challenge"

        def download(self, data_dir=None, cache_dir=None, download_mode=None):
            dataset = local_datasets["arc_challenge"]
            if dataset is None:
                return super().download(data_dir, cache_dir, download_mode)
            self.dataset = copy.deepcopy(dataset)

    local_task_classes = {
        "boolq": LocalBoolQ,
        "hellaswag": LocalHellaSwag,
        "arc_challenge": LocalARCChallenge,
    }
    resolved_tasks = []
    resolved_labels = []
    for task_name in task_names:
        task_cls = local_task_classes.get(task_name)
        dataset = local_datasets.get(task_name)
        if task_cls is not None and dataset is not None:
            resolved_tasks.append(task_cls())
            resolved_labels.append(f"{task_name}:local")
        else:
            resolved_tasks.append(task_name)
            resolved_labels.append(f"{task_name}:hub")
    return resolved_tasks, resolved_labels


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    resolved_tasks, resolved_labels = _resolve_eval_tasks(task_names)
    print(f"zero-shot task sources: {', '.join(resolved_labels)}")
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=resolved_tasks,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 