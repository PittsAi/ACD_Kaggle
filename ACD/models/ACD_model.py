import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel


class ACDModel(nn.Module):
    def __init__(self, args):
        super(ACDModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                              output_attentions=args.model.output_attentions,
                                              output_hidden_states=args.model.output_hidden_states)
        self.fc = nn.Linear(args.model.bert_hidden_size, args.model.num_class)
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

        if args.experiment.with_parameter_freeze:
            print(f'param.requires_grad:{args.experiment.with_parameter_freeze}')
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

    def forward(self, input_ids, input_mask, segment_ids, label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids)
        
        sequence_output = bert_output[1]
        # 通过一个线性层将第一个令牌的表示映射为池化输出
        pooled_output = self.bert.pooler.dense(sequence_output)
        # 应用激活函数到池化输出
        pooled_output = self.bert.pooler.activation(pooled_output)
        # 应用Dropout以防止过拟合
        pooled_output = self.dropout(pooled_output)
        # 通过线性分类器生成分类的logits
        output = self.fc(pooled_output)

        #output = self.fc(bert_output[1])
        output = self.softmax(output)
        loss = None

        if label is not None:
            loss = self.loss_fn(output, label)
            return loss, output

        return output