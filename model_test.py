import CaptionDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from nltk.translate.bleu_score import corpus_bleu

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
    return corpus_bleu(labels, predicts)

def calculate_score(batch_size = 2):
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


#test_model(1)
calculate_score(16)
