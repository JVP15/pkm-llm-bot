#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

torch.backends.cudnn.tf32 = True

seed = 42
transformers.set_seed(seed)



# In[2]:

if __name__ == '__main__':

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./models/gen9randombattle",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs/gen9randombattle",
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=50,
        save_steps=1000,
        tf32=True,
    )


    # In[3]:


    # Define the tokenizer and model
    model_name = 'gpt2' # w/ 2 epochs, normal got to ~.4, medium got to ~.35, large is getting to ~.3
    #model_name = 'meta-llama/Llama-2-7b-chat-hf'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to_bettertransformer()

    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=.1)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    chunk_size = 1024 # tokenizer.model_max_length


    # Define the dataset
    class ReplayDataset(torch.utils.data.Dataset):
        def __init__(self, replay_files, tokenizer):
            self.tokenizer = tokenizer
            self.replays = []

            # Load the replays
            for replay_file in tqdm(replay_files, total=len(replay_files)):
                with open(replay_file, "r", encoding='utf-8') as f:
                    replay = f.read()

                # Tokenize the replay
                tokenized_replay = tokenizer(replay)

                # Chunk the tokenized replay
                chunks = []
                for i in range(0, len(tokenized_replay["input_ids"]), chunk_size):
                    chunk = tokenized_replay["input_ids"][i:i + chunk_size]
                    chunks.append(chunk)

                # Add the chunks to the dataset
                for chunk in chunks:
                    self.replays.append(chunk)

        def __len__(self):
            return len(self.replays)

        def __getitem__(self, idx):
            return {'input_ids': self.replays[idx] }

    data_path = "dataset/gen9randombattle_rating/replays"
    replay_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    random.shuffle(replay_files)
    train_replay_files = replay_files[:int(len(replay_files) * 0.8)]
    val_replay_files = replay_files[int(len(replay_files) * 0.8):]

    train_dataset = ReplayDataset(train_replay_files, tokenizer)
    val_dataset = ReplayDataset(val_replay_files, tokenizer)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()


    # In[ ]:


    trainer.evaluate(val_dataset)


    # In[ ]:


    trainer.model = trainer.model.reverse_bettertransformer()
    #trainer.model.save_pretrained('models/gen9randombattle_gpt2_large')


    # In[ ]:


    # load the model
    model = AutoModelForCausalLM.from_pretrained('models/gen9randombattle_gpt2_large')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')


    # In[ ]:


    def generate_helper(input_text, **kwargs):
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        print(input_ids.shape)
        output = model.generate(input_ids, **kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)


    # In[ ]:


    input_text = """|turn|3
    |
    |t:|1699136500
    |move|p1a: Iron Leaves|Psyblade|p2a: Spidops
    |-damage|p2a: Spidops|0 fnt
    |faint|p2a: Spidops
    |-damage|p1a: Iron Leaves|175/275|[from] item: Life Orb
    |
    |upkeep
    |
    |t:|1699136505
    |switch|p2a: Salamence|Salamence, L77, M|273/273
    |-ability|p2a: Salamence|Intimidate|boost
    |-unboost|p1a: Iron Leaves|atk|1
    |turn|4
    |
    |t:|1699136509
    |move|p1a: Iron Leaves|Psyblade|p2a: Salamence
    |-damage|p2a: Salamence|35/273
    |-damage|p1a: Iron Leaves|148/275|[from] item: Life Orb
    |move|p2a: Salamence|Dual Wingbeat|p1a: Iron Leaves|[miss]
    |-miss|p2a: Salamence|p1a: Iron Leaves
    |
    |upkeep
    |turn|5
    |
    |t:|1699136518
    |move|p1a: Iron Leaves|Psyblade|p2a: Salamence
    |-damage|p2a: Salamence|0 fnt
    |faint|p2a: Salamence
    |-damage|p1a: Iron Leaves|121/275|[from] item: Life Orb
    |
    |upkeep
    |
    |t:|1699136524
    |switch|p2a: Basculin|Basculin-Blue-Striped, L87, F|264/264
    |turn|6
    |
    |t:|1699136528
    |move|p2a: Basculin|Wave Crash|p1a: Iron Leaves
    |-resisted|p1a: Iron Leaves
    |-damage|p1a: Iron Leaves|0 fnt
    |faint|p1a: Iron Leaves
    |-end|p1a: Iron Leaves|Quark Drive|[silent]
    |-damage|p2a: Basculin|224/264|[from] Recoil
    |
    |upkeep
    |
    |t:|1699136531
    |switch|p1a: Dragonite|Dragonite, L74, M|257/257
    |turn|7
    |
    |t:|1699136533
    |move|p2a: Basculin|Wave Crash|p1a: Dragonite
    |-resisted|p1a: Dragonite
    |-damage|p1a: Dragonite|189/257
    |-damage|p2a: Basculin|202/264|[from] Recoil
    |move|p1a: Dragonite|Dragon Dance|p1a: Dragonite
    |-boost|p1a: Dragonite|atk|1
    |-boost|p1a: Dragonite|spe|1
    |
    |upkeep
    |turn|8
    |
    |t:|1699136539
    |move|p1a:"""


    # In[ ]:


    generate_helper(input_text, max_new_tokens=16)


    # In[ ]:


    # look through all the files in gen9randombattle_rating/replays and see if any one of them has u-turn in it

