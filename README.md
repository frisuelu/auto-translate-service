# Translation of comments with open-source solution: Transformer architecture

Open-source solution to translate database comments into English. Although the only supported target language is English,
it's easy to adapt the code to be agnostic and allow for different input/output languages. The only limiting factor is the existence
of the model in the **Huggingface** library.

---

## Description

The files included are the following:

- `main.py`: standalone Python script for the comment translation. It will import the needed code from the rest of the files,
and perform translation over the annotation table where certain conditions are met (checking if the comment is already translated
and that the comment contains more than 2 words).

- `functions.py`: required auxiliary methods for the translation code. Two functions are defined: one to clean a string, and another
to split a sentence if the word amount is over a certin limit.

- `fasttext_transformer_ref.py`: this is the **main code** for the translation. It defines a class with two methods:

  - The first one (_get_languages_) uses Facebook's _fasttext_ library to detect the languages of the original comments before translation.
  - The second method (_transform_) translates the comments by: identifying their original language and loading the corresponding model
  for each one. An ordering of the comments based on their language is done first, so each model only has to be loaded once.

- `language_check.py`: script to check if the detected languages by `fasttext` match the language codes of the models in the HuggingFace library.
