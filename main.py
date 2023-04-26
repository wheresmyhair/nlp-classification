from torch.utils.data import DataLoader

from data_utils import create_examples
from tokenization import PretrainedTokenizer
from trainer import Trainer
from tqdm import tqdm

# Load tokenizer
tokenizer = PretrainedTokenizer(
    './models/tokenizer.model',
    max_seq_len=256,
    padding_strategy='max_length',
    pad_id=3
)

# Build DataLoader
train_dataset = create_examples('imdb', tokenizer, mode='train')
test_dataset = create_examples('imdb', tokenizer, mode='test')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Build Trainer
trainer = Trainer(
    num_cls      = 2,
    max_seq_len  = 256,
    d_model      = 512,
    n_layers     = 6,
    n_heads      = 8,
    dropout_rate = 0.8,
    d_ffn        = 2048,
    batch_size   = 128,
    lr           = 1e-4,
    train_loader = train_loader, 
    test_loader  = test_loader, 
    tokenizer    = tokenizer)

# Train & Validate
for epoch in tqdm(range(1, 51)):
    trainer.train(epoch)
    trainer.validate(epoch)
    trainer.save(epoch, 'testmodel', './models/transformer')
