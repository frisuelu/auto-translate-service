"""
File that includes functions needed for TeamEQ Transformer translation
"""
import re
import typing
import lxml.html
from lxml import etree


def clean_text(text: str) -> str:
    """
    Description
    -----------
    Clean text by removing emails and HTML tags. For HTML we use the lxml package.

    Arguments
    ---------
        - text (`str`): individual comment

    Returns:
        - `str`: clean comment
    """

    # Deleting email addresses
    text = re.sub("([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", "", text)

    # Remove HTML tags, but joins without spaces
    processed_text = lxml.html.document_fromstring(text)

    # Return the sub-string joined by a space
    return " ".join(etree.XPath("//text()")(processed_text))


def split_long_text(text: str, param: str = ".", size: int = 512) -> typing.List[str]:
    """
    Description
    -----------
    Split string based on a parameter at the end of each sentence.

    Arguments
    ---------
        - text (`str`): text to be split
        - param (`str`): parameter of the split function. Defaults to '.'.
        - size (`int`): maximum size of each split, throws warning if any
        split goes over this value. Defaults to 512.

    Returns:
        - return_strings `list(str)`: list of individual pieces of the comment
        padded to max length equal to `size` parameter
    """

    split = text.split(param)

    return_strings = []

    # Number of words per sentence
    words_per_line = [len(split[i].split()) for i in range(len(split))]

    # Check if any individual sentence is longer than size
    # In that case (for the time being) we only take the first (#size) words
    for pos, val in enumerate(words_per_line):
        if val > size:
            return_strings[pos] = split[pos].split()[:size]

    return return_strings
