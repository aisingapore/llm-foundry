import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from tokenizers import processors


from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}
SPIECE_UNDERLINE = "▁"

class SEABPETokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        legacy (`bool`, *optional*, defaults to `True`):
            Whether or not the `legacy` behaviour of the tokenizer should be used. Legacy is before the merge of #24622
            which includes fixes to properly handle tokens that appear after special tokens. 
            legacy means we are not modifying existing tokenizers without knowing. (And we need to manually update those core tokenizers)
            
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import T5Tokenizer

            >>> tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)
            >>> tokenizer.encode("Hello <extra_id_0>.")
            [8774, 32099, 3, 5, 1]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import T5Tokenizer

            >>> tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
            >>> tokenizer.encode("Hello <extra_id_0>.")  # the extra space `[3]` is no longer here
            [8774, 32099, 5, 1]
            ```
            Checkout the pull request and the issue [here](https://github.com/huggingface/transformers/pull/24565) for
            more details.

    """

    def __init__(
        self,
        vocab_file,
        # unk_token="<unk>",
        # bos_token=None,
        eos_token="<|endoftext|>",
        # pad_token="<|padding|>",
        unk_token=None,
        bos_token=None,
        # eos_token=None,
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        legacy=None,
        **kwargs,
    ):

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        print('1 in token_SEA_BPE, kwargs:', self.sp_model_kwargs)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        print('1 in token_SEA_BPE, sp_model is init as:', self.sp_model)
        self.sp_model.Load(vocab_file)
        print('1 eos_token', eos_token)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            legacy=legacy,
            **kwargs,
        )
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This means that tokens that come after special tokens will not be properly handled. We recommend you to"
                " read the related pull request available at https://github.com/huggingface/transformers/pull/24565, and set the legacy attribute accordingly."
            )
            legacy = True

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        # print('2 in token_SEA_BPE, kwargs:', self.sp_model_kwargs)
        # self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # print('2 in token_SEA_BPE, sp_model is init as:', self.sp_model)
        # self.sp_model.Load(vocab_file)
        # print('2 eos_token', eos_token)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text, **kwargs) -> List[str]:
        # Replace the SPIECE_UNDERLINE with a space to make sure SPIECE_UNDERLINE is only used at
        # the beginning of the text
        if not self.legacy:
            text = SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " ")
        return super().tokenize(text, **kwargs)

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text):
        """
        Returns a tokenized string.

        Since the sentencepiece internal model always adds a SPIECE_UNDERLINE, at the beginning of the provided text,
        we need to remove it by hand when the current text is a subsequence. This happens whenever the `self.tokenize`
        function is called with specials tokens: the input is split on the special tokens, and each subsequence is
        passed to `_tokenize`. Thus if a subsequence did not start with a `" "` or SPIECE_UNDERLINE, we have to remove
        the extra `SPIECE_UNDERLINE` prepended.
        """
        if not self.legacy:
            is_first = text.startswith(SPIECE_UNDERLINE)
            if is_first:
                text = text[1:]

        tokens = self.sp_model.encode(text, out_type=str)

        if not self.legacy and not is_first and not text.startswith(" ") and tokens[0].startswith(SPIECE_UNDERLINE):
            tokens = ([tokens[0][1:]] if len(tokens[0]) > 1 else []) + tokens[1:]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string
    
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)