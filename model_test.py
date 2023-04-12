import CaptionDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from nltk.translate.bleu_score import corpus_bleu
import evaluate
import pickle

from tqdm import tqdm

def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-trained")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    return model, tokenizer

def convert_labels(tensors, tokenizer):
    for tensor in tensors:
        tensor[tensor == -100] = tokenizer.pad_token_id
    
    return tensors


def get_bleu_score(predicts:list, labels:list):
    labels_formatted = [[x] for x in labels]
    return corpus_bleu(labels_formatted, predicts)

def get_meteor_score(predicts:list, labels:list):
    meteor = evaluate.load('meteor')
    labels_formatted = [[x] for x in labels]
    
    results = meteor.compute(predictions=predicts, references=labels_formatted)
    return results['meteor']

def get_rouge_L_score(predicts:list, labels:list):
    rouge = evaluate.load('rouge')
    labels_formatted = [[x] for x in labels]

    results = rouge.compute(predictions=predicts, references=labels_formatted)
    return results['rougeL']



def calculate_score(batch_size = 2, score_function = get_bleu_score):
    model, tokenizer = get_model()
    dataset = CaptionDataset.get_hf_ds()
    tokenized_ds = CaptionDataset.tokenize_ds(dataset, tokenizer, deep_copy=True)
   
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    _, _, test_dataloader = CaptionDataset.get_hf_dataLoaders(tokenized_ds, collator, train_batch=batch_size, val_batch=batch_size, test_batch=batch_size)


    labels_lst =[]
    predicts_lst = []
    for batch in tqdm(test_dataloader):
        labels_tensor = convert_labels(batch['labels'], tokenizer)

        labels = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
        
        outputs = model.generate(**batch)
        predicts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels_lst.extend(labels)
        predicts_lst.extend(predicts)
    
    return predicts_lst, labels_lst#score_function(predicts_lst, labels_lst)

        

def test_model(batch_size = 2):
    model, tokenizer = get_model()
    dataset = CaptionDataset.get_hf_ds()
    tokenized_ds = CaptionDataset.tokenize_ds(dataset, tokenizer, deep_copy=True)
    print(tokenized_ds)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    train_dataloader, val_dataloader, test_dataloader = CaptionDataset.get_hf_dataLoaders(tokenized_ds, collator, train_batch=batch_size, val_batch=batch_size, test_batch=batch_size)
     
    for batch in test_dataloader:
        break
    outputs = model.generate(**batch)
    #print(outputs.logits)
    print(batch['labels'])
    print(convert_labels(batch['labels'], tokenizer))
    print(outputs)
    print(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print(tokenizer.batch_decode(convert_labels(batch['labels'], tokenizer), skip_special_tokens=True))


if __name__ == '__main__':
    #test_model(1)
    #preds, labels = calculate_score(32)
    # with open('preds.pkl', 'wb') as f:
    #    pickle.dump(preds, f)
    # with open('labels.pkl', 'wb') as f:
    #    pickle.dump(labels, f)
    dataset = CaptionDataset.get_hf_ds()
    print(dataset)
    with open('preds.pkl', 'rb') as f:
        preds = pickle.load(f)
    with open('labels.pkl', 'rb') as file:
        labels = pickle.load(file)
    print(len(preds), len(labels))
    print(get_bleu_score(preds, labels))
    print(get_meteor_score(preds, labels))
    print(get_rouge_L_score(preds, labels))
    


