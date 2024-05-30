# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
# from transformers.generation_utils import top_k_top_p_filtering
from transformers.generation.utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,)
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput,)
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            # labels[:, 0] = -100

            # my
            labels[labels>=self.toker.vocab_size] = -100
            
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        if validation:
            original_lm_logits_size = lm_logits.size()
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        masked_lm_loss = None
        if labels is not None:
            # if lm_logits.size(-1) != (labels.max() + 1):
            # if validation:
                 # print("lm_logits: {}\tlabels: {}".format(lm_logits.size(), labels.max() + 1))
                # print("original_lm_logits: {}\ttoker.vocab_size: {}".format(original_lm_logits_size, self.toker.vocab_size))
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else: # validation
            assert not self.training
            return loss, label_size
        
    def predict_emotion(self, logits, encoded_info):
        assert not self.training
        # print(encoded_info)
        # print("predict logits", logits.size())

        user_id=None
        sys_id=None
        logits_user = logits[:, 0, -7:]
        logits_sys = logits[:, 0, self.toker.vocab_size+10:-8]

        if user_id is not None:
            pred_user = user_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits_user / TEMPERATURE, top_p=0.9)
                pred_user = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred_user = torch.argmax(logits_user, dim=-1)
        
        pred_user_top1 = torch.topk(logits_user, k=1, dim=-1)[1]

        if sys_id is not None:
            pred_sys = sys_id
        else:
            if SAMPLE:
                filtered_logits_sys = top_k_top_p_filtering(logits_sys / TEMPERATURE, top_p=0.9)
                pred_sys = torch.multinomial(F.softmax(filtered_logits_sys, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred_sys = torch.argmax(logits_sys, dim=-1)
        
        pred_sys_top1 = torch.topk(logits_sys, k=1, dim=-1)[1]

        if encoded_info is not None:
            encoded_info.update({
                'pred_user_id': pred_user, 
                'pred_user_id_top1': pred_user_top1,
                'pred_user_id_dist': F.softmax(logits_user, dim=-1),
                'pred_sys_id': pred_sys, 
                'pred_sys_id_top1': pred_sys_top1,
                'pred_sys_id_dist': F.softmax(logits_sys, dim=-1),
            })


    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        if encoded_info is not None:
            strat_id = encoded_info.get('strat_id', None)
        else:
            strat_id=None

        # logits = logits[:, 0, -8:]

        logits = logits[:, 0, self.toker.vocab_size:self.toker.vocab_size+10]
    
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)
        
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        # encoded_info.update({
        #     'pred_strat_id': pred,
        #     'pred_strat_id_top1': pred_top1,
        #     'pred_strat_id_top3': pred_top3,
        #     'pred_strat_id_dist': F.softmax(logits, dim=-1)
        # })

        if encoded_info is not None:
            encoded_info.update({
                'pred_strat_id': pred,
                'pred_strat_id_top1': pred_top1,
                'pred_strat_id_top3': pred_top3,
                'pred_strat_id_dist': F.softmax(logits, dim=-1)
            })
        # else:
        #     # 如果 encoded_info 是空值，可以选择创建一个新的字典并更新
        #     encoded_info = {
        #         'pred_strat_id': pred,
        #         'pred_strat_id_top1': pred_top1,
        #         'pred_strat_id_top3': pred_top3,
        #         'pred_strat_id_dist': F.softmax(logits, dim=-1)
        #     }
            
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        user_id = None,
        strat_id = None,
        sys_id = None,
        other_res = None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = other_res
        # encoded_info = kwargs.pop('encoded_info', None) 
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        # print(lm_logits.size())
        self.predict_strategy(lm_logits, encoded_info)
        self.predict_emotion(lm_logits, encoded_info)
        
        if encoded_info is not None:
            # print(encoded_info)
           
            # print(decoder_input_ids)
            sep_emotion = torch.tensor([54961], device=decoder_input_ids.device)
            sep_emotion = sep_emotion = sep_emotion.unsqueeze(1)
            sep_emotion = sep_emotion.expand(decoder_input_ids.size(0), decoder_input_ids.size(1))

            pred_user_id = encoded_info['pred_user_id'][..., None] + 54962
            pred_sys_id = encoded_info['pred_sys_id'][..., None]+54954
            pred_strat_id = encoded_info['pred_strat_id'][..., None] + 54944
            # print(decoder_input_ids)
            # print(pred_sys_id)
            # print(pred_user_id)
            # print(pred_strat_id)
            # print((pred_sys_id + len(self.toker) - 25).size())
     
            decoder_input_ids = torch.cat([decoder_input_ids, pred_user_id,sep_emotion,pred_strat_id,
                                           pred_sys_id], dim=-1)

            # print("decoder input", decoder_input_ids)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]