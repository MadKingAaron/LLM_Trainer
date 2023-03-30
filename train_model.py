import CaptionDataset
import train_test_model


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to('cuda')

df = pd.read_csv('./trainVal_samples.csv')


trainloader, valloader, testloader = CaptionDataset.get_loaders(tokenizer, 'Predict next step in sequence:\n', prepend_prefix=True,
                                                                df=df, train_split=.6, train_batch=64, val_split=0.3, val_batch=64,
                                                                test_split=0.1, test_batch=64)

optimizer, scheduler = train_test_model.get_optimzer(initial_lr=5e-8, model=model)

model = train_test_model.train_model(optimizer=optimizer, trainloader=trainloader, valloader=valloader, model=model, epochs=10, scheduler=scheduler, device='cuda', tb_comment='LR_5e-8_Epoch_10')

model.save_pretrained('./flan-t5-small-trained', from_pt=True)
