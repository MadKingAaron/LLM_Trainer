import CaptionDataset
import train_test_model


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import pandas as pd
import check_gpu_mem


def train():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    df = pd.read_csv('./trainVal_samples.csv')


    trainloader, valloader, testloader = CaptionDataset.get_loaders(tokenizer, 'Predict next step in sequence:\n', prepend_prefix=True,
                                                                    df=df, train_split=.6, train_batch=64, val_split=0.3, val_batch=64,
                                                                    test_split=0.1, test_batch=64)

    optimizer, scheduler = train_test_model.get_optimzer(initial_lr=5e-8, model=model)

    model = train_test_model.train_model(optimizer=optimizer, trainloader=trainloader, valloader=valloader, model=model, epochs=10, scheduler=scheduler, device='cuda', tb_comment='LR_5e-8_Epoch_10')
    model.save_pretrained('./flan-t5-small-trained', from_pt=True)


def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    return model, tokenizer


def get_best_device(num_devices:int=8):
    best_dev = check_gpu_mem.get_best_free_gpu(num_devices)
    return best_dev

def train_hf_ds(batch_size=64):
    model, tokenizer = get_model()

    device = get_best_device(8)
    #device = -1
    print('GPU:', device)
    model = model.to(device)
    
    optimizer, lr_scheduler = train_test_model.get_optimzer(initial_lr=3e-8, model=model)

    dataset = CaptionDataset.get_hf_ds()
    tokenized_ds = CaptionDataset.tokenize_ds(dataset, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    train_dataloader, val_dataloader, test_dataloader = CaptionDataset.get_hf_dataLoaders(tokenized_ds, collator, train_batch=batch_size, val_batch=batch_size, test_batch=batch_size)

    model = train_test_model.train_model_hf(optimizer, trainloader=train_dataloader, valloader=val_dataloader, model=model, epochs=50, scheduler=lr_scheduler,
                                                   tb_comment="FLAX_T5_LR_3e-8_EPOCH_50", device=device)
    model.save_pretrained('./flan-t5-small-trained', from_pt=True)

if __name__ == "__main__":
    train_hf_ds()