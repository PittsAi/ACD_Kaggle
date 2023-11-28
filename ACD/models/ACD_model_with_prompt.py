import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel

class ACDModelWithPrompt(nn.Module):
    def __init__(self, args):
        super(ACDModelWithPrompt, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                              output_attentions=args.model.output_attentions,
                                              output_hidden_states=args.model.output_hidden_states)

        self.fc = nn.Linear(args.model.bert_hidden_size, args.model.num_class)
        self.embeddings = self.bert.embeddings
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

        if args.experiment.with_parameter_freeze:
            print(f'param.requires_grad:{args.experiment.with_parameter_freeze}')
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        self.pre_seq_len = args.model.pre_seq_len
        self.prefix_tokens = torch.arange(self.args.model.pre_seq_len).long()
        # self.prefix_tokens = nn.Parameter(torch.randn(args.model.pre_seq_len, args.model.bert_hidden_size))
        self.prefix_encoder = torch.nn.Embedding(self.args.model.pre_seq_len, args.model.bert_hidden_size)

    def get_prompt(self, batch_size):
        # prompts = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(self.bert.device)
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
        self, 
        input_ids, 
        input_mask, 
        segment_ids, 
        label=None, 
        inputs_embeds=None, 
        position_ids=None, 
        token_type_ids=None, 
        output_attentions=None,
        output_hidden_states=None):

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        # inputs_embeds = torch.cat((raw_embedding, prompts), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        input_mask = torch.cat((prefix_attention_mask, input_mask), dim=1)
        # print(f'input_mask:{input_mask.shape}')
        # print(f'inputs_embeds:{inputs_embeds.shape}') 
        if torch.cuda.device_count() > 1:
            self.bert = torch.nn.DataParallel(self.bert)
            outputs = self.bert.module(
                attention_mask=input_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,           
            )
        else:
            outputs = self.bert(
                attention_mask=input_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,           
            )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]

        pooled_output = self.bert.pooler.dense(first_token_tensor)
        pooled_output = self.bert.pooler.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        #output = self.fc(first_token_tensor)
        output = self.softmax(output)
        loss = None

        if label is not None:
            loss = self.loss_fct(output, label)
            return loss, output

        return output