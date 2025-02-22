from pathlib import Path
from tree_sitter import Language, Parser, Node
import functools

import numpy as np
from numpy.random import RandomState

from typing import List, Tuple, Optional

Language.build_library(
    f"{Path(__file__).parent}/build/languages.so",
    [f"{Path(__file__).parent}/tree-sitter-typescript/typescript"]
)
TS_LANGUAGE = Language(
    f"{Path(__file__).parent}/build/languages.so", 'typescript')
PARSER = Parser()
PARSER.set_language(TS_LANGUAGE)


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD = (
            f"<fim_suffix>",
            f"<fim_prefix>",
            f"<fim_middle>",
            f"<fim_pad>",
        )

        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id

def get_prefix_middle_suffix(np_rng: RandomState, sample: bytes, strip_suffix_rate: float) -> Optional[Tuple[Tuple[str, str, str], RandomState]]:
    def is_child_type_annotation(node):
        """Checks if any of the parent nodes is an annotation node."""
        node = node.parent
        while node is not None:
            if node.type == "type_annotation" or node.type == "opting_type_annotation" or node.type == "omitting_type_annotation":
                return True
            node = node.parent
        return False

    def contains_url(node):
        # check if it contains a url, should not contain a //
        string = sample[node.start_byte:node.end_byte].decode("utf-8")
        return "//" in string

    QUERY = TS_LANGUAGE.query("""
[
  (type_annotation) @annotation
  (opting_type_annotation) @annotation
  (omitting_type_annotation) @annotation
]
""")
    tree = PARSER.parse(sample)

    # Each capture has a start_byte and end_byte; these are the indices of the
    # type annotation. We want to invert these indices, i.e. get the substrings
    # between the captures (and also the substring before the first capture and
    # the substring after the last capture).
    captures: List[Tuple[Node, str]] = QUERY.captures(tree.root_node)

    def is_splitable(node):
        return not is_child_type_annotation(node) and not contains_url(node)

    def is_capturable(node):
        return not is_child_type_annotation(node) and not contains_url(node)

    captures_no_child: List[Node] = []

    for i, (node, _) in enumerate(captures):
        if is_capturable(node):
            captures_no_child += [node]

    splittable_indices: List[int] = []

    for i, node in enumerate(captures_no_child):
        if is_splitable(node):
            splittable_indices += [i]

    if len(splittable_indices) == 0:
        return None
    random_pick_i = np_rng.choice(splittable_indices)

    prefix_b: bytes = sample[:captures_no_child[random_pick_i].start_byte]
    middle_b: bytes = sample[captures_no_child[random_pick_i]
                             .start_byte:captures_no_child[random_pick_i].end_byte]

    if middle_b.startswith(b":"):
        prefix_b += b": "
        middle_b = middle_b[1:].lstrip()
    suffix_b: bytes = b""

    # if we strip the types to the suffix:
    if np_rng.binomial(1, strip_suffix_rate):
        l = len(captures_no_child)
        for i in range(random_pick_i, l - 1):
            suffix_b += sample[captures_no_child[i]
                               .end_byte:captures_no_child[i + 1].start_byte]
        suffix_b += sample[captures_no_child[l - 1].end_byte:]
    else:  # keep the types in the suffix
        suffix_b = sample[captures_no_child[random_pick_i].end_byte:]

    prefix_str = prefix_b.decode("utf-8")
    middle_str = middle_b.decode("utf-8")
    suffix_str = suffix_b.decode("utf-8")

    return (prefix_str, middle_str, suffix_str), np_rng


# Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    tokenizer,
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    strip_suffix_rate=0.9,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    if np_rng.binomial(1, fim_rate):
        decoded_bytes: str | bytes = tokenizer.decode(sample)
        if not isinstance(decoded_bytes, bytes):
            decoded_bytes = decoded_bytes.encode("utf-8")

        try:
            res = get_prefix_middle_suffix(
                np_rng, decoded_bytes, strip_suffix_rate)

        except Exception as e:
            print(e)
            print("GOT FAILED SAMPLE:\n", decoded_bytes)
            return None, np_rng

        if res is None:
            return None, np_rng

        (prefix_str, middle_str, suffix_str), np_rng = res

        prefix = np.array(tokenizer.encode(prefix_str))
        middle = np.array(tokenizer.encode(middle_str))
        suffix = np.array(tokenizer.encode(suffix_str))

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng


if __name__ == "__main__":  # some unit tests
    import os
    rng = np.random.RandomState(seed=int(os.urandom(4).hex(), 16))
    sample = """
    interface Foo {
        foo(x: number, y: number): number;
        name: {
            first: string;
            last: {
                name: string;
                age: number;
            };
        };
    }

    function foo(x: number, y:number):number {
        return x + y;
    }

    // some unicode to mess things up
    // 😀 😃 😄 😁 😆 😅

    function foo2(x:number, y: number): number {
        return x + y;
    }

    interface Bar {
        bar(x: number, y: number): number;
        name: {
            first: string;
            last: string;
        };
    }
                

    function url() {
        let url = `https://127.0.0.1:${SUBSTRATE_PORT}`
    }
    """
    bytes_sample = bytes(sample, "utf-8")
    print("sample:", sample)
    print("bytes_sample:", bytes_sample)

    print("get_prefix_middle_suffix:")
    res = get_prefix_middle_suffix(rng, bytes_sample, 0.5)
    if res is not None:
        (prefix_str, middle_str, suffix_str), rng = res
        print("prefix_str:", prefix_str)
        print("middle_str:", middle_str)
        print("suffix_str:", suffix_str)
