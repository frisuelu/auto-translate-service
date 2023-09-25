"""
This script contains the code from the reference for the automatic detection
of the language + batch translations for fast execution and multiple language
translation
(https://towardsdatascience.com/translate-any-two-languages-in-60-lines-of-python-b54dc4a9e739).
"""

from typing import List, Optional, Set
import fasttext
from transformers import MarianTokenizer, MarianMTModel
import os
import requests
import functions
import tqdm


class LanguageTransformerFast:
    """
    Description
    -----------
    Translate a list of texts from one language to another. Defaults to a
    target language of English.

    A total of 1336 (as of 2021-10-22) translation models are available
    (https://huggingface.co/Helsinki-NLP). Language code names may be
    inconsistent; most of them can be checked here
    (https://huggingface.co/transformers/v4.0.1/model_doc/marian.html?highlight=mariantokenizer#naming),
    and codes like code_region (e.g, es_AR for Argentina's Spanish) would not
    need to be used.

    The set all_languages was obtained from here
    (https://developers.google.com/admin-sdk/directory/v1/languages) removing
    the region coded languages as well as the ones with no translation to
    English.

    Default usage will take into account that ANY language could show, but
    maybe not all of them can be translated. In that case, the transform()
    method will report the detected language and add it to the final results.
    If the model for that language doesn't exist/can't be imported, then we
    will register the code (at least) and not translate it. In any other case,
    translations should be done.

    Methods
    -------

        - get_language(): use the FastText library from Facebook to detect the
        language of each comment. Check the detectable language list here
        (https://fasttext.cc/docs/en/language-identification.html)
        - transform(): order comments based on their language, translate them
        (accounting for the max length) and reorder them to fit the previous
        order

    Some models can translate multiple languages at once. These are recorded
    here:

        - (https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en): for
        Spanish, Italian, French, Romansh, Romanian, Galician, Corsican,
        Walloon, Portuguese, Occitan, Aragonese, Indonese, Haitian Creole,
        Latin...
        - (https://huggingface.co/Helsinki-NLP/opus-mt-sla-en): for Slavic
        languages (Belarusian, Croatian, Macedonian, Czech, Russian, Polish,
        Bulgarian, Ukranian, Slovenian).
        - (https://huggingface.co/Helsinki-NLP/opus-mt-bat-en): for Baltic
        languages (Lithuanian, Latvian, Latgalian, Samogitian).
        - (https://huggingface.co/Helsinki-NLP/opus-mt-mul-en): lots of
        languages, but may compromise performance of translation.

    For the rest, if they're not contained in a group we will use the
    `all_languages` set to load the individual models. The following languages
    are not supported as of yet (translating into English):

        - Norwegian (no): no connection to English
        (only in the "Helsinki-NLP/opus-mt-mul-en" model)
        - Serbian (sr): no model (either single or multiple language)

    Greek (el) is supported through another model
    (https://huggingface.co/lighteternal/SSE-TUC-mt-el-en-cased, and
    Hebrew (he) is supported through another model as well
    (https://huggingface.co/tiedeman/opus-mt-he-en). Both are also supported
    through the "Helsinki-NLP/opus-mt-mul-en" model.
    """

    def __init__(
        self,
        fasttext_model_path: str = "./tmp/lid.176.bin",
        allowed_langs: Optional[Set[str]] = None,
        target_lang: str = "en",
        log: bool = False,
    ):
        self.fasttext_model_path = fasttext_model_path

        # The allowed langs are the languages that WILL be translated; if a
        # language is not included, it will be detected (by its lang_code) but
        # not translated
        self.allowed_langs = allowed_langs

        self.target_lang = target_lang
        self.romance_langs = {
            "it",
            "es",
            "fr",
            "pt",
            "oc",
            "ca",
            "rm",
            "ro",
            "wa",
            "gl",
            "an",
            "lld",
            "fur",
            "lij",
            "lmo",
            "frp",
            "lad",
            "mwl",
            "co",
            "nap",
            "scn",
            "vec",
            "sc",
            "la",
        }
        self.baltic_langs = {"lt", "lv"}
        self.slavic_langs = {"be", "hr", "mk", "cs", "ru", "pl", "bg", "uk", "sl"}
        # List of language codes that load individual models
        self.individual_langs = {
            "ar",
            "eu",
            "bn",
            "da",
            "nl",
            "et",
            "fi",
            "de",
            "hi",
            "hu",
            "is",
            "id",
            "ja",
            "ko",
            "ml",
            "mr",
            "zh",
            "sk",
            "sw",
            "sv",
            "th",
            "tr",
            "ur",
            "vi",
            "cy",
        }

        # Take all available languages
        # Hebrew and Greek are separate since they load different models
        if allowed_langs is None:
            self.allowed_langs = self.romance_langs.union(
                self.slavic_langs,
                self.baltic_langs,
                self.individual_langs,
                {"el", "he", "no"},
            )

        else:
            self.allowed_langs = allowed_langs

        self.log = log

    def get_language(self, texts: List[str]) -> List[str]:
        """
        Description
        -----------
        Detect the comment language using the `fasttext` library from Facebook.

        Arguments
        ---------
            - texts (`list(str)`): list of comments to analyze

        Returns:
            - `list(str)`: list of character codes representing the detected
            languages
        """
        # If the model doesn't exist download it
        # Two options exist: lid.176.bin (larger, more accurate), and
        # lid.176.ftz (smaller & compressed)
        if not os.path.isfile(self.fasttext_model_path):
            url = (
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
            r = requests.get(url, allow_redirects=True)
            open("./tmp/lid.176.bin", "wb").write(r.content)

        lang_model = fasttext.load_model(self.fasttext_model_path)

        # If an empty string appears, don't try to execute the replace code as
        # it will fail
        langs = []
        for text in texts:
            if isinstance(text, str):
                langs.append(" ")
            elif text.isspace():
                langs.append(" ")
            else:
                text = text.replace("\n", " ")
                lang, _ = lang_model.predict(text)
                langs.append(lang)

        # Extract the two character language code from the predictions.
        return [x[0].split("__")[-1] for x in langs]

    def transform(self, texts: List[str]) -> List[List[str]]:
        """
        Description
        -----------
        Transform the comments using the required HuggingFace model. The model
        is selected based on the detected language using the `fasttext` library.

        Arguments
        ---------
            - texts (`list(str)`): list of comments to analyze

        Returns:
            - `list(list(str))`: each list value contains
        """
        # Get the language codes for each text in texts
        langs = self.get_language(texts)

        # Zip the texts, languages, and their indecies
        # sort on the language so that all languages appear together
        text_lang_pairs = sorted(
            zip(texts, langs, range(len(langs))), key=lambda x: x[1]
        )
        model = None

        # CLI logging of the progress when the flag is True
        if self.log:
            looper = tqdm.tqdm(text_lang_pairs)
        else:
            looper = text_lang_pairs

        translations = []
        prev_lang = text_lang_pairs[0]

        for text, lang, idx in looper:
            lang_tmp = lang  # We save the actual prediction to append it later

            # If the detected language is the target, don't translate
            if lang == self.target_lang:
                translations.append((idx, text, lang_tmp, text))

            # If the language is not allowed, add an empty string
            elif lang not in self.allowed_langs:
                translations.append((idx, text, lang_tmp, ""))

            else:
                # ROMANCE model
                if lang in self.romance_langs and self.target_lang == "en":
                    author = "Helsinki-NLP"
                    model_name = "opus-mt-ROMANCE-en"

                # Slavic model
                elif lang in self.slavic_langs and self.target_lang == "en":
                    author = "Helsinki-NLP"
                    model_name = "opus-mt-sla-en"

                # Baltic model
                elif lang in self.baltic_langs and self.target_lang == "en":
                    author = "Helsinki-NLP"
                    model_name = "opus-mt-bat-en"

                # Individual model
                elif lang in self.individual_langs and self.target_lang == "en":
                    author = "Helsinki-NLP"
                    model_name = f"opus-mt-{lang}-{self.target_lang}"

                # Greek model (we instantiate it inside so it doesn't
                # overwrite later)
                elif lang == "el" and self.target_lang == "en":
                    author = "lighteternal"
                    model_name = "SSE-TUC-mt-el-en-cased"

                # Hebrew model (we instantiate it inside so it doesn't
                # overwrite later)
                elif lang == "he" and self.target_lang == "en":
                    author = "tiedeman"
                    model_name = "opus-mt-he-en"

                # Norwegian model (probably not performant)
                elif lang == "no" and self.target_lang == "en":
                    author = "Helsinki-NLP"
                    model_name = "opus-mt-mul-en"

                else:
                    raise ValueError(
                        f"Found unsupported language code {lang}, aborting..."
                    )

                # TODO: SUPPORT FOR SERBIAN IS NOT POSSIBLE RIGHT NOW

                # Load new models when language changes
                if model is None or lang != prev_lang:
                    translation_model_name = f"{author}/{model_name}"

                    # Download the model and tokenizer
                    model = MarianMTModel.from_pretrained(translation_model_name)
                    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

                # Clean the text
                text = functions.clean_text(text)

                # Check number of words
                words = len(text.split())

                # Context size is 512; if more words are present we
                # need to split the comment in sentences
                if words > 512:
                    split_comment = functions.split_long_text(text)

                    # Translate each sentence separately
                    for i, _ in enumerate(split_comment):
                        placeholder = []
                        inputs = tokenizer(
                            split_comment[i], return_tensors="pt", padding=True
                        )
                        gen = model.generate(**inputs)
                        placeholder.append(
                            functions.clean_text(
                                tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                            )
                        )

                    # Join them as one string later on
                    translations.append([idx, text, lang_tmp, " ".join(placeholder)])
                    prev_lang = lang

                # Regular comment
                else:
                    # Tokenize the text
                    batch = tokenizer(text, return_tensors="pt", padding=True)

                    gen = model.generate(**batch)
                    translations.append(
                        (
                            idx,
                            text,
                            lang_tmp,
                            functions.clean_text(
                                tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                            ),
                        )
                    )
                    prev_lang = lang

        # Reorganize the translations to match the original ordering
        return [x[1:] for x in sorted(translations, key=lambda x: x[0])]
