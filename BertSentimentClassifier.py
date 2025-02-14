from transformers import AutoTokenizer, BertModel
import torch
from torch import nn

class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes, PRE_TRAINED_MODEL):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        '''_, cls_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
        )'''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Intentamos usar pooler_output, pero si es None, tomamos el CLS token de last_hidden_state
        cls_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output