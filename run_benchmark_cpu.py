import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime import SessionOptions
import time

def sentiment_analysis_onnx_batched(model_id, df, field_name, batch_size, threads):
    file_name = "onnx/model.onnx"

    #set the number of threads
    options = SessionOptions()
    options.intra_op_num_threads = threads

    model = ORTModelForSequenceClassification.from_pretrained(model_id, file_name=file_name, provider="CPUExecutionProvider", session_options = options)
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

def sentiment_analysis_batched(model_id, df, field_name, batch_size, threads):
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(f'cpu')
    torch.set_num_threads(threads)

    start_time = time.time()
    results_df = pd.DataFrame()
    for start_idx in tqdm(range(0, len(df), batch_size), desc=f"(Batch size {batch_size})"):
        end_idx = start_idx + batch_size
        texts = df[field_name].iloc[start_idx:end_idx].tolist()

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to(f'cpu')
        attention_mask = inputs['attention_mask'].to(f'cpu')

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
    parser = argparse.ArgumentParser(description="Benchmark CPUs running BERT code")
    parser.add_argument("--dataset", type=str, choices=["normal", "filtered"], default="normal", help="Dataset to use")
    parser.add_argument("--model", type=str, choices=["pytorch", "onnx"], required=True, help="Model to use")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads to use")
    parser.add_argument("--batches", type=str, default="1,2,4", help="Comma-separated batch sizes to run")

    args = parser.parse_args()

    field_name = "body"
    model_id = "SamLowe/roberta-base-go_emotions"
    model_id_onnx = "SamLowe/roberta-base-go_emotions-onnx"

    if args.dataset == "filtered":
        str_dataset = 'data/random_sample_10k_filtered.csv.gz'
    else:
        str_dataset = 'data/random_sample_10k.csv.gz'

    df = pd.read_csv(str_dataset, compression='gzip')

    results = {}
    batch_sizes = [int(x) for x in args.batches.split(',')]
    for batch_size in batch_sizes:
        if args.model == "onnx":
            elapsed_time, messages_per_second = sentiment_analysis_onnx_batched(model_id_onnx, df, field_name, batch_size=batch_size, threads=args.threads)
        elif args.model == "pytorch":
            elapsed_time, messages_per_second = sentiment_analysis_batched(model_id, df, field_name, batch_size=batch_size, threads=args.threads)
   
        results[batch_size] = {'elapsed_time': elapsed_time, 'messages_per_second': messages_per_second}

    print("\nDataset: {}, Model: {}, CPU threads: {}\n".format(args.dataset, args.model, args.threads))
    print("size\tmessages/s\tmessages/s/thread")
    for batch_size, result in results.items():
        print(f"{batch_size}\t{result['messages_per_second']:.2f}\t\t{result['messages_per_second']/args.threads:.2f}")
