import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_uniform_
from transformers import EncoderDecoderModel, GPT2Tokenizer, GPT2Model, ViltProcessor, ViltModel
from PIL import Image

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TF_Classifier(nn.Module):
    def __init__(self, args, vocab_size):
        super(TF_Classifier, self).__init__()
        self.embed_dim = 768
        self.vocab_size = vocab_size
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim) #vocaburaly embedding
        self.position_encoding = PositionalEncoding(768)

        # TF layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=4, batch_first=False)
        self.tf_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.softmax = nn.Softmax(dim=1)

        self.generator = nn.Linear(768, vocab_size)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights() # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1,0.1)

        self.generator.bias.data.fill_(0)
        self.generator.weight.data.uniform_(-0.1,0.1)

    def forward(self, args, last_hidden_states, tf_cls_tgt_input):
        '''
        tgt – the sequence to the decoder (required).

        memory – the sequence from the last layer of the encoder (required).

        tgt_mask – the mask for the tgt sequence (optional).

        memory_mask – the mask for the memory sequence (optional).

        tgt_key_padding_mask – the mask for the tgt keys per batch (optional).

        memory_key_padding_mask – the mask for the memory keys per batch (optional).
        '''
        last_hidden_states = last_hidden_states.permute(1,0,2) #(ipt_len, batch, feature_dim)
        #print('(MODEL) last_hidden_states:', last_hidden_states.shape)

        tgt = tf_cls_tgt_input.permute(1,0) # (tgt_len, batch)
        tgt_length = tgt.size(0)

        tgt_mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1) #(tgt_len, tgt_len)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        #print('(MODEL) tgt_embedding:', tgt_embedding.shape, tgt_embedding)
        tgt_embedding = self.position_encoding(tgt_embedding) #(length, batch, feature_dim)

        pred = self.tf_decoder(tgt=tgt_embedding, memory=last_hidden_states, tgt_mask=tgt_mask) #(length, batch, feature_dim)
        #print('(MODEL) pred:', pred.shape, pred)

        pred = self.generator(self.dropout(pred)) #(length, batch, vocab_size)
        #print('(MODEL) pred:', pred.shape, pred)
        pred = pred.permute(1,0,2) #(batch, length, vocab_size)
        #print('(MODEL) pred:', pred.shape, pred)

        return pred


class Middle_TF(nn.Module):
    def __init__(self):
        super(Middle_TF, self).__init__()

        self.embed_dim = 768

        # TF layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.softmax = nn.Softmax(dim=1)

        self.generator = nn.Linear(768, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights() # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.generator.bias.data.fill_(0)
        self.generator.weight.data.uniform_(-0.1,0.1)

    def forward(self, last_hidden_states):

        tf_pred = self.tf_encoder(src=last_hidden_states) #(batch, ipt_len, feature_dim)
        #print('(MODEL) pred:', pred.shape, pred)

        tf_cls_feature = tf_pred[:, 0, :] #(batch, feature_dim)
        tf_cls_pred = self.generator(self.dropout(tf_pred)) #(batch, 2)

        #print('(MODEL) pred:', pred.shape, pred)

        return tf_pred, tf_cls_pred



class Vilt_GPT2(nn.Module):
    def __init__(self, args, gpt2_tokenizer, tf_vocab_size=None):
        super(Vilt_GPT2, self).__init__()

        self.vocab_size = len(gpt2_tokenizer)
        self.tokenizer = gpt2_tokenizer

        vilt_model = "dandelin/vilt-b32-mlm"
        gpt2_model = args.gpt2_model

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(vilt_model, gpt2_model)

        if args.decoder_classify_bias:
            embedding_layer = self.model.decoder.resize_token_embeddings(len(gpt2_tokenizer))  # Update the model embeddings with the new vocabulary size

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        if args.only_crossattention:
            for name, param in self.model.named_parameters():
                if "crossattention" not in name:
                    param.requires_grad = False
        elif args.freeze_gpt2:
            for name, param in self.model.named_parameters():
                if "crossattention" not in name and "encoder" not in name:
                    param.requires_grad = False
        elif args.freeze_vilt:
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False

        if args.encoder_classify_bias or args.only_encoder_classify_bias:
            #self.classifier = nn.Linear(768, 2)
            mlp = []
            mlp.append(nn.Linear(768, 768*2, bias=True))
            mlp.append(nn.LeakyReLU())
            mlp.append(nn.Linear(768*2, 2, bias=True))
            self.mlp = nn.Sequential(*mlp)
            self.softmax = nn.Softmax(dim=1)
        elif args.tf_bias_classify:
            self.tf_classifier = TF_Classifier(args, tf_vocab_size)


    def forward(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels, tf_cls_tgt_input):

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                            pixel_values=pixel_values, pixel_mask=pixel_mask, output_attentions=True, output_hidden_states=True)

        if args.encoder_classify_bias or args.only_encoder_classify_bias:
            last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)
            cls_feature = last_hidden_states[:, 0, :] #(batch, feature_dim)
            logits = self.mlp(cls_feature) #(batch, 2)
        else:
            logits = None

        if args.tf_bias_classify:
            last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)
            tf_pred = self.tf_classifier(args, last_hidden_states, tf_cls_tgt_input)
        else:
            tf_pred = None

        ### Decoder ###
        outputs = self.model(encoder_outputs=encoder_outputs, labels=labels)
        loss = outputs.loss

        return loss, outputs, logits, tf_pred

    def generate_text(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels, tf_cls_tgt_input):
        ##generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        ##                                    pixel_values=pixel_values, pixel_mask=pixel_mask, decoder_start_token_id=50256)

        ##generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            pixel_values=pixel_values, pixel_mask=pixel_mask)
        if args.encoder_classify_bias or args.only_encoder_classify_bias:
            last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)
            cls_feature = last_hidden_states[:, 0, :] #(batch, feature_dim)
            logits = self.mlp(cls_feature) #(batch, 2)
            preds = self.softmax(logits)
            tf_pred = None
        elif args.tf_bias_classify:
            last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)
            tf_pred = self.tf_classifier(args, last_hidden_states, tf_cls_tgt_input)
            preds = None
        else:
            preds = None
            tf_pred = None


        ### Decoder ###
        beam_outputs = self.model.generate(encoder_outputs=encoder_outputs, max_length=args.decoder_max_len, min_length=3, num_beams=args.num_beams, do_sample=args.do_sample, no_repeat_ngram_size=args.no_repeat_ngram_size, num_return_sequences=1, early_stopping=True, repetition_penalty=args.repetition_penalty)

        ##generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        ##for i, beam_output in enumerate(beam_outputs):
        ##    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

        return beam_outputs, preds, tf_pred




class Vilt_TF_GPT2(nn.Module):
    def __init__(self, args, gpt2_tokenizer):
        super(Vilt_TF_GPT2, self).__init__()

    
        self.vocab_size = len(gpt2_tokenizer)
        self.tokenizer = gpt2_tokenizer

        self.middle_tf = Middle_TF()

        vilt_model = "dandelin/vilt-b32-mlm"
        gpt2_model = args.gpt2_model

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(vilt_model, gpt2_model)

        if args.decoder_classify_bias:
            embedding_layer = self.model.decoder.resize_token_embeddings(len(gpt2_tokenizer))  # Update the model embeddings with the new vocabulary size

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        if args.freeze_gpt2:
            for name, param in self.decoder.named_parameters():
                if "crossattention" not in name and "encoder" not in name:
                    param.requires_grad = False
        elif args.freeze_vilt:
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False


    def forward(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels):

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            pixel_values=pixel_values, pixel_mask=pixel_mask, output_attentions=True, output_hidden_states=True)

        last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)

        ### Middle TF Encoder ###
        last_hidden_states, tf_cls_pred = self.middle_tf(last_hidden_states) #(batch, ipt_len, feature_dim), (batch, ipt_len, 2)

        ### Decoder ###
        outputs = self.model(encoder_outputs=(last_hidden_states), labels=labels)
        loss = outputs.loss

        return loss, outputs, tf_cls_pred
        
    def generate_text(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels):

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            pixel_values=pixel_values, pixel_mask=pixel_mask)

        last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)

        ### Middle TF Encoder
        last_hidden_states, tf_cls_pred = self.middle_tf(last_hidden_states) #(batch, ipt_len, feature_dim), (batch, ipt_len, 2)

        ### Decoder ###
        beam_outputs = self.model.generate(encoder_outputs=(last_hidden_states), max_length=args.decoder_max_len, num_beams=args.num_beams, do_sample=args.do_sample, no_repeat_ngram_size=args.no_repeat_ngram_size, num_return_sequences=1, early_stopping=False, repetition_penalty=args.repetition_penalty)

        return beam_outputs, tf_cls_pred









class Vilt_GPT2_with_decoipt(nn.Module):
    def __init__(self, args, gpt2_tokenizer):
        super(Vilt_GPT2_with_decoipt, self).__init__()

        self.vocab_size = len(gpt2_tokenizer)
        self.tokenizer = gpt2_tokenizer

        vilt_model = "dandelin/vilt-b32-mlm"
        gpt2_model = args.gpt2_model

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(vilt_model, gpt2_model)

        embedding_layer = self.model.decoder.resize_token_embeddings(len(gpt2_tokenizer))  # Update the model embeddings with the new vocabulary size

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size


        if args.only_crossattention:
            for name, param in self.model.named_parameters():
                if "crossattention" not in name:
                    param.requires_grad = False

        #self.generator = nn.Linear(feature_dim, vocab_size)

    def forward(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels):

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            pixel_values=pixel_values, pixel_mask=pixel_mask, output_attentions=True, output_hidden_states=True)

        ### Decoder ###
        outputs = self.model(encoder_outputs=encoder_outputs, decoder_input_ids=tokenized_captions["input_ids"],
                            decoder_attention_mask=tokenized_captions["attention_mask"], labels=labels, return_dict=True)
        loss = outputs["loss"]

        return loss, outputs

    def generate_text(self, args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, tokenized_captions, labels):
        ##generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        ##                                    pixel_values=pixel_values, pixel_mask=pixel_mask, decoder_start_token_id=50256)

        ##generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        ### Encoder ###
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            pixel_values=pixel_values, pixel_mask=pixel_mask)
        ##last_hidden_states = encoder_outputs.last_hidden_state #(batch, ipt_len, feature_dim)
        ##last_hidden_states = last_hidden_states.permute(1,0,2) #(ipt_len, batch, feature_dim)
        #print('(Model) last_hidden_states.shape:', last_hidden_states.shape)
        #print('(Model) last_hidden_states.device:', last_hidden_states.device)

        ### Decoder ###
        beam_outputs = self.model.generate(encoder_outputs=encoder_outputs, max_length = args.decoder_max_len, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 1, early_stopping = True)

        ##generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        ##for i, beam_output in enumerate(beam_outputs):
        ##    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

        return beam_outputs

