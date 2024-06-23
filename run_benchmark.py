import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import time

def sentiment_analysis_batched(df, device_type, model_type, model_id, batch_size, field_name, file_name = None, gpu_id=0, num_threads=1):

    if device_type == 'gpu':
        device = torch.device(f'cuda:{gpu_id}')
    elif device_type == 'cpu':
        device = torch.device('cpu')
        torch.set_num_threads(num_threads)

    if model_type == 'torch':
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)

    elif model_type == 'onnx':
        if device_type == 'cpu':
            model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name, provider="CPUExecutionProvider")
        elif device_type == 'gpu':
            model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name, provider="CUDAExecutionProvider", provider_options={'device_id': gpu_id})

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    start_time = time.time()
    results = []

    # Precompute id2label mapping
    id2label = model.config.id2label

    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"(Batch size {batch_size})"):
        end_idx = start_idx + batch_size
        texts = df[field_name].iloc[start_idx:end_idx].tolist()

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.sigmoid(outputs.logits)  # Use sigmoid for multi-label classification

        # Collect predictions on GPU
        results.append(predictions)

    # Concatenate all results on GPU
    all_predictions = torch.cat(results, dim=0).cpu().numpy()

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions, columns=[id2label[i] for i in range(all_predictions.shape[1])])

    # Add prediction columns to the original DataFrame
    combined_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    elapsed_time = time.time() - start_time
    messages_per_second = len(df) / elapsed_time

    return elapsed_time, messages_per_second, combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPUs running BERT code")
    parser.add_argument("--dataset", type=str, choices=["normal", "filtered"], default="normal", help="Dataset to use (normal or filtered)")
    parser.add_argument("--model", type=str, choices=["torch", "onnx", "onnx-fp16"], help="Model to use (torch, onnx or onnx-fp16)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], help="Device to use (cpu or gpu)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default 0)")
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32", help="Comma-separated batch sizes to run")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads to use (default 1)")

    args = parser.parse_args()

    # Models
    model_ids = {
        "torch": {'model_type': 'torch', 'model_id': "SamLowe/roberta-base-go_emotions", 'file_name': None},
        "onnx": {'model_type': 'onnx', 'model_id': "SamLowe/roberta-base-go_emotions-onnx", 'file_name': "onnx/model.onnx"},
        "onnx-fp16": {'model_type': 'onnx', 'model_id': "joaopn/roberta-base-go_emotions-onnx-fp16", 'file_name': "model.onnx"}
    }


    if args.model == "onnx-fp16" and args.device == "cpu":
        raise ValueError("ONNX FP16 models are only supported on GPUs")

    field_name = "body"
    model = args.model
    model_params = model_ids[model]

    if args.dataset == "filtered":
        str_dataset = 'data/random_sample_10k_filtered.csv.gz'
    else:
        str_dataset = 'data/random_sample_10k.csv.gz'

    df = pd.read_csv(str_dataset, compression='gzip')


    results = {}
    batch_sizes = [int(x) for x in args.batches.split(',')]
    for batch_size in batch_sizes:

        elapsed_time, messages_per_second,_ = sentiment_analysis_batched(df, args.device, model_params['model_type'], model_params['model_id'], batch_size, field_name, model_params['file_name'], gpu_id=0, num_threads=args.threads)

        results[batch_size] = {'elapsed_time': elapsed_time, 'messages_per_second': messages_per_second}

    if args.device == 'gpu':
        gpu_name = torch.cuda.get_device_name(args.gpu)
        print("\nDataset: {}, Model: {}, GPU: {}\n".format(args.dataset, args.model, gpu_name))
        print("size\tmessages/s")
        for batch_size, result in results.items():
            print(f"{batch_size}\t{result['messages_per_second']:.2f}")
    else:
        print("\nDataset: {}, Model: {}, CPU threads: {}\n".format(args.dataset, args.model, args.threads))
        print("size\tmessages/s\tmessages/s/thread")
        for batch_size, result in results.items():
            print(f"{batch_size}\t{result['messages_per_second']:.2f}\t\t{result['messages_per_second']/args.threads:.2f}")
