from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from datasets import load_dataset
import pandas as pd
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

# custom_data = [
# 	{"text" : "[ intervention ] used for treating { condition }",
# 	"label" : 1
# 	},
# 	{"text" : "[ intervention ] led to a severe case of { condition }",
# 	"label" : 0
# 	},
# 	{"text" : "[ intervention ] improved outcomes in patients with { condition }",
# 	"label" : 1
# 	},
# 	{"text" : "Patients with { condition } treated with [ intervention ] developed cough ",
# 	"label" : 1
# 	},
# 	{"text" : "Patients with atrial fibrillation treated with [ intervention ] developed { condition } ",
# 	"label" : 0
# 	}
# ]

custom_data = pd.DataFrame([['[ intervention ] used for treating { condition }', 1],
	['[ intervention ] led to a severe case of { condition }', 0],
	['[ intervention ] improved outcomes in patients with { condition }', 1],
	['Patients with { condition } treated with [ intervention ] developed cough', 1],
	['Patients with atrial fibrillation treated with [ intervention ] developed { condition } ', 0],
	['Patient with { condition } who received [ intervention ] had a significant improvement in mortality ', 1]],
	columns=['text', 'label'])

dataset = Dataset.from_pandas(custom_data)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# print(dataset[0])


def preprocess_function(examples):
	return tokenizer(examples["text"], truncation=True)

tokenized = dataset.map(preprocess_function, batched=True)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_set = tokenized.to_tf_dataset(
	columns=["attention_mask", "input_ids", "label"],
	shuffle=True,
	batch_size=16,
	collate_fn=data_collator,)


# optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

# model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# model.compile(optimizer=optimizer, loss=None)
# model.fit(x=tf_train_set, epochs=num_epochs)

# imdb = load_dataset("imdb")


# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# def preprocess_function(examples):
# 	return tokenizer(examples["text"], truncation=True)

# tokenized_imdb = imdb.map(preprocess_function, batched=True)



# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# tf_train_set = tokenized_imdb['train'].to_tf_dataset(columns=["attention_mask", "input_ids", "label"],
# 	shuffle=True,
# 	batch_size=16,
# 	collate_fn=data_collator,)

# tf_validation_set = tokenized_imdb["test"].to_tf_dataset(

#     columns=["attention_mask", "input_ids", "label"],

#     shuffle=False,

#     batch_size=16,

#     collate_fn=data_collator,

# )

batch_size = 2
num_epochs = 5

batches_per_epoch = len(tokenized) // batch_size

total_train_steps = int(batches_per_epoch * num_epochs)


optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.compile(optimizer=optimizer)

model.fit(x=tf_train_set, epochs=num_epochs)