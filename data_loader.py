import transformers
from datasets import load_dataset, load_metric

datasets = load_dataset("csv", data_files="medium-articles.zip")

# create train / validation 
datasets_train_test = datasets["train"].train_test_split(test_size=3000)
datasets_train_validation = train_test["train"].train_test_split(test_size=3000)

datasets["train"] = datasets_train["train"]
datasets["validation"] = datasets_val["test"]
datasets["test"] = datasets_test["test"]
