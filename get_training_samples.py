import CaptionDataset
from random import sample
import random
import os
import json

from tqdm import tqdm
import pandas as pd

random.seed(1234)

PROMPT_FOLDER = './gpt_prompts'
GT_FILE = './gpt_gt/ground_truth.json'


#dataset = CaptionDataset.get_hf_ds()
#examples = sample(list(dataset['train']), 10)

#val_ex = sample(list(dataset['validation']), 1)

prefix = 'Predict next step in the sequence for the recipe:\n'

# formatted_samples = ['Given the recipe is '+str(x['recipe_type'])+'. '+prefix+str(x['input']) for x in examples]
# labels = [x['label'] for x in examples]

# united_out = ['Input: '+ x[0] + "\n" + "Output: "+ x[1] + "\n" for x in zip(formatted_samples, labels)]


def format_example(examples:list):
    return ['Input: Given the recipe is '+str(x['recipe_type'])+'. '+prefix+str(x['input']) + "\nOutput: " + x['label'] + '\n' for x in examples]



def get_recipe_examples(ds:list, recipe_type:str, max_samples:int):
    
    examples = [x for x in ds if x['recipe_type'] == recipe_type]
    examples =  list(sample(examples, max_samples)) if len(examples) > max_samples else examples
    return format_example(examples)


def write_prompt_to_file(examples:list, question:str, ground_truth:str, file = None):
    print("You're job is to predict the next step in a sequence of steps, given a specific task. That is to say that given a task such as a cooking, you must predict the next step in the recipe, given the steps fed to you.\nBelow are sample inputs and outputs:\n", file=file)

    full_output = "You're job is to predict the next step in a sequence of steps, given a specific task. That is to say that given a task such as a cooking, you must predict the next step in the recipe, given the steps fed to you.\nBelow are sample inputs and outputs:\n\n"
    for x in examples:
        print(x, file=file)
        full_output += (x + '\n')
    
    print('\n\nPredict the following:\n', file=file)
    full_output += '\n\nPredict the following:\n\n'
    print(question, file=file)
    print('Output: ', file=file)
    full_output += question
    full_output += '\nOutput: '


    #print('\n\n\n\nGT:', ground_truth, file=file)

    file.close()

    return full_output

def generate_question(prompt:dict):
    return 'Input: Given the recipe is '+str(prompt['recipe_type'])+'. '+prefix+str(prompt['input'])


def write_prompts(examples:list, test:list, num_prompts:int):
    prompts = sample(test, num_prompts)
    labels = []
    questions = []
    full_prompts = []
    for i in tqdm(range(num_prompts)):
        prompt = prompts[i]
        filename = 'prompt_'+str(i)+'.txt'

        prompt_examples = get_recipe_examples(examples, prompt['recipe_type'], 12)


        question = generate_question(prompt)
        questions.append(question)

        ground_truth = prompt['label'] 
        labels.append(ground_truth)

        file = open(os.path.join(PROMPT_FOLDER, filename), 'w')

        full_output = write_prompt_to_file(prompt_examples, question, ground_truth, file)
        full_prompts.append(full_output)


    with open(GT_FILE, 'w') as gt_file:
        questions_and_labels = {'question': questions, 'label':labels, 'full_prompt':full_prompts, 'predicts':([None] * len(questions))}
        json.dump(questions_and_labels, gt_file)
    
    return pd.DataFrame(questions_and_labels)


#print(united_out)

# file = open('./prompt.txt', 'w')

# print("You're job is to predict the next step in a sequence of steps, given a specific task. That is to say that given a task such as a cooking, you must predict the next step in the recipe, given the steps fed to you.\nBelow are sample inputs and outputs:\n", file=file)
# for x in united_out:
#     print(x, file=file)


# print('\n\nPredict the following:\n', file=file)
# x = val_ex[0]
# val_input = 'Given the recipe is '+str(x['recipe_type'])+'. '+prefix+str(x['input'])
# print('Input:', val_input, file=file)


if __name__ == "__main__":
    examples = list(CaptionDataset.get_hf_ds()['train'])
    test = list(CaptionDataset.get_hf_ds()['test'])
    df = write_prompts(examples, test, 300)
    df.to_excel('./prompts.xlsx', engine='openpyxl', index=False)