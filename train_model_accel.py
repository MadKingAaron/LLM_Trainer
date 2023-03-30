import torch
import CaptionDataset
import train_test_model_accel

from accelerate import Accelerator, find_executable_batch_size
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq


def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    return model, tokenizer


def training_function(train_batch_size):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=train_batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator # Ensure they can be used in our context
        
        accelerator.free_memory() # Free all lingering references
        
        model, tokenizer = get_model()
        
        model.to(accelerator.device)
        
        optimizer, lr_scheduler = train_test_model_accel.get_optimzer(initial_lr=3e-8, model=model)
        
        dataset = CaptionDataset.get_hf_ds()
        tokenized_ds = CaptionDataset.tokenize_ds(dataset, tokenizer)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        train_dataloader, val_dataloader, test_dataloader = CaptionDataset.get_hf_dataLoaders(tokenized_ds, collator, train_batch=batch_size, val_batch=batch_size, test_batch=batch_size)
        
        
        
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        model = train_test_model_accel.train_model(optimizer, accelerator, trainloader=train_dataloader, valloader=val_dataloader, model=model, epochs=50, scheduler=lr_scheduler,
                                                   tb_comment="FLAX_T5_LR_3e-8_EPOCH_50")
        model.save_pretrained('./flan-t5-small-trained', from_pt=True)

    inner_training_loop()


if __name__ == "__main__":
    training_function(train_batch_size=64)
