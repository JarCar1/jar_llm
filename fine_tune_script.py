import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def main():
    model_name = 'mosaicml/mpt-1b-redpajama-200b-dolly'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token

    def tokenize_function(example):
        model_inputs = tokenizer(example['instruction'], truncation=True, padding='max_length')
        model_inputs = {key: torch.tensor(val) for key, val in model_inputs.items()}
        if 'attention_mask' in model_inputs:
            model_inputs['attention_mask'] = model_inputs['attention_mask'].bool()
        return model_inputs

    dataset = load_dataset("NumbersStation/NSText2SQL")
    dataset = dataset.filter(lambda example: example['source'] != 'spider')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./english_to_sql_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10,
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps=10_000,
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()

if __name__ == "__main__":
    main()
