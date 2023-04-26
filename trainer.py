from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer.model import TransformerEncoder

class Trainer:
    def __init__(
        self, 
        num_cls,
        max_seq_len, 
        d_model, 
        n_layers,
        n_heads, 
        dropout_rate, 
        d_ffn, 
        batch_size,
        lr,
        train_loader, 
        test_loader, 
        tokenizer
    ):
        self.num_cls = num_cls
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.d_ffn = d_ffn
        self.batch_size = batch_size
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = TransformerEncoder(vocab_size  = self.vocab_size,
                                        seq_len     = self.max_seq_len,
                                        num_cls     = self.num_cls,
                                        d_model     = self.d_model,
                                        n_layers    = self.n_layers,
                                        n_heads     = self.n_heads,
                                        p_drop      = self.dropout_rate,
                                        d_ff        = self.d_ffn,
                                        pad_id      = self.pad_id)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            # |inputs| : (batch_size, seq_len), |labels| : (batch_size)

            outputs, attention_weights = self.model(inputs)
            # |outputs| : (batch_size, 2), |attention_weights| : [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
            
            loss = self.criterion(outputs, labels)
            losses += loss.item()
            acc = (outputs.argmax(dim=-1) == labels).sum()
            accs += acc.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:4f}%'.format(
                    i, i, n_batches, losses/i, accs/(i*self.batch_size)*100.))

        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses/n_batches, accs/n_samples*100.))
            
    def validate(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                inputs, labels = map(lambda x: x.to(self.device), batch)
                # |inputs| : (batch_size, seq_len), |labels| : (batch_size)

                outputs, attention_weights = self.model(inputs)
                # |outputs| : (batch_size, 2), |attention_weights| : [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
                
                loss = self.criterion(outputs, labels)
                losses += loss.item()
                acc = (outputs.argmax(dim=-1) == labels).sum()
                accs += acc.item()

        print('Valid Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses/n_batches, accs/n_samples*100.))

    def save(self, epoch, model_prefix='model', root='models'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.model, path)


# #### TEST MODULE ####
# from torch.utils.data import DataLoader

# from data_utils_self import create_examples
# from tokenization_self import PretrainedTokenizer
# tokenizer = PretrainedTokenizer(
#     './models/tokenizer.model',
#     max_seq_len=128,
#     padding_strategy='max_length',
#     pad_id=3
# )
# model = TransformerEncoder(
#     vocab_size  = 130344,
#     seq_len     = 128,
#     num_cls     = 2,
#     d_model     = 512,
#     n_layers    = 6,
#     n_heads     = 8,
#     p_drop      = 0.2,
#     d_ff        = 2048,
#     pad_id      = 3)

# model.to('cuda')

# optimizer = optim.Adam(model.parameters(), 0.01)
# criterion = nn.CrossEntropyLoss()

# train_dataset = create_examples('imdb', tokenizer, mode='train')
# test_dataset = create_examples('imdb', tokenizer, mode='test')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# losses, accs = 0, 0
# n_batches, n_samples = len(train_loader), len(train_loader.dataset)

# model.train()
# for i, batch in enumerate(train_loader):
#     inputs, labels = map(lambda x: x.to('cuda'), batch)
#     # |inputs| : (batch_size, seq_len), |labels| : (batch_size)

#     outputs, attention_weights = model(inputs)
#     # |outputs| : (batch_size, 2), |attention_weights| : [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
    
#     loss = criterion(outputs, labels)
#     losses += loss.item()
#     acc = (outputs.argmax(dim=-1) == labels).sum()
#     accs += acc.item()
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if i % (n_batches//5) == 0 and i != 0:
#         print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:4f}%'.format(
#             i, i, n_batches, losses/i, accs/(i*self.batch_size)*100.))

# print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses/n_batches, accs/n_samples*100.))
# #####################