import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
try:
    from openai.openai_object import OpenAIObject
except Exception:
    class OpenAIObject(dict):
        def to_dict_recursive(self):
            return dict(self)
import copy




import os, requests, time

def _ollama_generate(model, prompt, temperature=0.7, top_p=1.0, max_tokens=1024, stop=None, host=None):
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        }
    }
    if stop:
        payload["stop"] = stop
    r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # normalize to repo's expected shape
    return {"text": data.get("response",""), "finish_reason": "stop"}


StrOrOpenAIObject = Union[str, OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(prompts, model_name, batch_size, decoding_args, logit_bias=None):
    # decoding_args: has .temperature, .top_p, .max_tokens, .n, .stop
    temperature = getattr(decoding_args, "temperature", 0.7)
    top_p = getattr(decoding_args, "top_p", 1.0)
    max_tokens = getattr(decoding_args, "max_tokens", 1024)
    stop = getattr(decoding_args, "stop", None)

    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        for p in batch:
            res = _ollama_generate(
                model=model_name,
                prompt=p,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            results.append(res)
            time.sleep(0.05)
    return results


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
