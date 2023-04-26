import torch
from tokenization import PretrainedTokenizer

# Load tokenizer
tokenizer = PretrainedTokenizer(
    './models/tokenizer.model', 
    max_seq_len=256,
    padding_strategy='max_length', 
    pad_id=3
)

# Load model
model = torch.load('./models/transformer/best.pt').to('cuda')
model.eval()

# Make input
text = 'what a bad movie'
input_ids = tokenizer.encode([text])[0]
input_ids = torch.tensor([input_ids]).to('cuda')

# Inference
output, attention_weights = model(input_ids)
print('class: {}'.format(output.argmax(dim=1)))