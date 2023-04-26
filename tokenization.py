from typing import List, Optional, Union
import sentencepiece as spm

class TextTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.vocab_size()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]):
        return self.sp.DecodeIds(ids)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self.sp.DecodePieces(tokens)

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)

    def __len__(self):
        return self.vocab_size


class PretrainedTokenizer:
    def __init__(self, vocab_file: str,max_seq_len: int = 128, padding_strategy: str = None, pad_id: int = None):
        self.tokenizer = TextTokenizer(vocab_file)
        self.max_seq_len = max_seq_len
        self.padding_strategy = padding_strategy
        self.pad_id = pad_id
        self.max_blank_length = 80
        self.vocab_size = self.tokenizer.vocab_size
        if self.padding_strategy is not None:
            assert self.pad_id is not None, "pad_id must be specified when padding_strategy is not None."

    def _get_text_tokenizer(self):
        return self.tokenizer
    
    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        return f"<|tab|>"
    
    @property
    def num_tokens(self):
        return self.tokenizer.vocab_size

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", PretrainedTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, PretrainedTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, self.max_blank_length)
        return text

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) in an id (integer).
        """
        return self.tokenizer.convert_token_to_id(token)

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id (integer) in a token (str).
        """
        return self.tokenizer.convert_id_to_token(id)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens in list of ids.
        """
        return [self.convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(id) for id in ids]

    def _pad(
            self,
            encoded_inputs: List[List[int]],
            max_seq_len,
            pad_id: int,
            padding_strategy: str = None
    ):
        '''
        ## Description
        - Batch padding
        '''
        assert padding_strategy in ["max_length", "longest", None]
        assert isinstance(encoded_inputs[0], list)

        if padding_strategy is None:
            return encoded_inputs

        if padding_strategy == "max_length":
            for i, encoded_input in enumerate(encoded_inputs):
                encoded_inputs[i] = self._pad_single(
                    encoded_input, max_seq_len, pad_id, padding_strategy)
            return encoded_inputs

        elif padding_strategy == "longest":
            pass

    def _pad_single(
            self,
            encoded_input: List[int],
            max_seq_len,
            pad_id: int,
            padding_strategy: str = None
    ):
        assert padding_strategy in ["max_length", "longest"]

        if padding_strategy == "max_length":
            difference = max_seq_len - len(encoded_input)
            encoded_input = encoded_input + [pad_id] * difference
            return encoded_input

        elif padding_strategy == "longest":
            raise NotImplementedError



    def encode(
            self, text: List[str], linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[int]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        assert isinstance(text, list)
        for i, content in enumerate(text):
            tmp = self._preprocess(content, linebreak, whitespaces)
            if not add_dummy_prefix:
                tmp = "<n>" + tmp
            tmp = self._get_text_tokenizer().encode(tmp)
            tmp = tmp[2:] if not add_dummy_prefix else tmp
            tmp = tmp[:self.max_seq_len]
            text[i] = tmp
        tokens = self._pad(text, self.max_seq_len, self.pad_id, self.padding_strategy)
        return tokens

    def decode(self, text_ids: List[int]) -> str:
        # TODO: Change for pad
        ids = [int(_id) for _id in text_ids]
        text = self._get_text_tokenizer().decode(ids)
        text = text.replace("<n>", "\n")
        text = text.replace(PretrainedTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def tokenize(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[str]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer().tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]
    

# #### TEST MODULE ####
# tkn = PretrainedTokenizer(
#     './models/tokenizer.model', 
#     max_seq_len=10, 
#     padding_strategy='max_length',
#     pad_id=0)
# a = tkn.encode(['你\n好', 'world'])

tkn = TextTokenizer('./models/tokenizer.model')
tkn.sp.vocab_size()
# tkn.convert_id_to_token(3)