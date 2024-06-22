import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import time

def sentiment_analysis_onnx_batched(model_id, file_name, df, field_name, batch_size, gpu_id):
    
    model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name, provider="CUDAExecutionProvider", provider_options={'device_id': gpu_id})
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Function to classify emotions of multiple texts in batched mode and return scores
    def classify_texts(texts):
        # Tokenize the batch of texts
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)

        probabilities = torch.sigmoid(outputs.logits)
        labels = model.config.id2label  # Adjust if necessary
        
        # Process each item in the batch
        batch_results = []
        for prob in probabilities:
            result = {labels[i]: prob_item.item() for i, prob_item in enumerate(prob.squeeze())}
            batch_results.append(result)
            
        return batch_results

    start_time = time.time()

    # Placeholder for aggregated results
    results = []
    
    # Iterate through the DataFrame in batches
    for start in tqdm(range(0, len(df), batch_size), desc=f"(batch size {batch_size})"):
        end = start + batch_size
        batch_texts = df[field_name].iloc[start:end].tolist()
        
        # Apply classify_text to the entire batch at once
        batch_results = classify_texts(batch_texts)
        
        # Assuming batch_results is a list of dictionaries
        results.extend(batch_results)
        
    # Merge the results with the original DataFrame
    results_df = pd.DataFrame(results, index=df.index[:len(results)])
    df = pd.concat([df, results_df], axis=1)

    elapsed_time = time.time() - start_time
    messages_per_second = len(df) / elapsed_time

    return elapsed_time, messages_per_second

def sentiment_analysis_batched(model_id, df, field_name, batch_size, gpu_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(f'cuda:{gpu_id}')

    start_time = time.time()
    results_df = pd.DataFrame()
    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"(Batch size {batch_size})"):
        end_idx = start_idx + batch_size
        texts = df[field_name].iloc[start_idx:end_idx].tolist()

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to(f'cuda:{gpu_id}')
        attention_mask = inputs['attention_mask'].to(f'cuda:{gpu_id}')

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        batch_results = [{'label': model.config.id2label[prediction.argmax().item()], 'score': prediction.max().item()} for prediction in predictions]
        batch_results_df = pd.DataFrame(batch_results, index=range(start_idx, start_idx + len(batch_results)))
        results_df = pd.concat([results_df, batch_results_df], axis=0)

    elapsed_time = time.time() - start_time
    messages_per_second = len(df) / elapsed_time

    return elapsed_time, messages_per_second

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GPUs running BERT code")
    parser.add_argument("--dataset", type=str, choices=["normal", "filtered"], default="normal", help="Dataset to use")
    parser.add_argument("--model", type=str, choices=["pytorch", "onnx", "onnx-fp16"], default="pytorch", help="Model to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32", help="Comma-separated batch sizes to run")

    args = parser.parse_args()

    # Models
    model_id = "SamLowe/roberta-base-go_emotions"

    model_id_onnx = "SamLowe/roberta-base-go_emotions-onnx"
    file_name_onnx = "onnx/model.onnx"
    
    model_id_onnx_fp16 = "joaopn/roberta-base-go_emotions-onnx-fp16"
    file_name_onnx_fp16 = "model.onnx"

    field_name = "body"

    if args.dataset == "filtered":
        str_dataset = 'data/random_sample_10k_filtered.csv.gz'
    else:
        str_dataset = 'data/random_sample_10k.csv.gz'

    df = pd.read_csv(str_dataset, compression='gzip')

    results = {}
    batch_sizes = [int(x) for x in args.batches.split(',')]
    for batch_size in batch_sizes:
        if args.model == "onnx":
            elapsed_time, messages_per_second = sentiment_analysis_onnx_batched(model_id_onnx, file_name_onnx, df, field_name, batch_size=batch_size, gpu_id=args.gpu)
        elif args.model == "pytorch":
            elapsed_time, messages_per_second = sentiment_analysis_batched(model_id, df, field_name, batch_size=batch_size, gpu_id=args.gpu)
        elif args.model == "onnx-fp16":
            elapsed_time, messages_per_second = sentiment_analysis_onnx_batched(model_id_onnx_fp16, file_name_onnx_fp16, df, field_name, batch_size=batch_size, gpu_id=args.gpu)
   
        results[batch_size] = {'elapsed_time': elapsed_time, 'messages_per_second': messages_per_second}

    gpu_name = torch.cuda.get_device_name(args.gpu)
    print("\nDataset: {}, Model: {}, GPU: {}\n".format(args.dataset, args.model, gpu_name))
    print("size\tmessages/s")
    for batch_size, result in results.items():
        print(f"{batch_size}\t{result['messages_per_second']:.2f}")
