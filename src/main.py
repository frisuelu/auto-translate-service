"""
Translation script with logic needed for final use. Uses the
`fasttext_transformer_ref` script, imports the class
`LanguageTransformerFast()` and applies the .transform() method to translate
the comments.

Current logic takes into account the "translation" column of the "annotation"
table that contains the comments in the database. When a new row is inserted,
the associated column should be empty; in that case, the script runs
translation over the new row and appends the result to the column.

The logic should either be:

    - Creating a new column, "translation", in the annotation table and
    inserting an empty string in the column when a new comment is created.
    - Creating a new table where comments are aggregated and translated with
    the same logic. This should help with multiple read/write operations, if
    comments are being added and translated at the same time.

Another simpler option is available: when new rows are appended, they are
inserted in a CSV and added to a path; the script could be listening and just
translate everything and export it somewhere.
"""

import pandas as pd
import numpy as np
from fasttext_transformer_ref import LanguageTransformerFast
from sqlalchemy import create_engine


def main(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Description
    -----------
    Run translation on comments inside a DataFrame. An expected table structure
    is set, but it could easily be changed if we decided to use another table.

    We take advantage of the batch nature of the translator; in another case,
    maybe we would iterate over all rows, check if NAs are present, and then
    translate each comment individually.
    """

    # We will use multiprocessing for parallelization
    import multiprocessing as mp

    # Obtain the rows that need translation
    rows_to_translate = dataframe[
        (dataframe["lang_code"].isna()) & (dataframe["translation"].isna())
    ]

    # Remove from translation: non-strings & comments with less than 3 words
    good_rows = [
        len(rows_to_translate.loc[idx, "annotation"].split()) >= 3
        if isinstance(rows_to_translate.loc[idx, "annotation"], str)
        else False
        for idx in rows_to_translate.index
    ]

    rows_to_translate = rows_to_translate[good_rows]

    # List of lists containing original comment, language code and translation
    translator = LanguageTransformerFast(log=True)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        dataframe.loc[
            rows_to_translate.index, ["annotation", "lang_code", "translation"]
        ] = pool.apply_async(
            translator.transform, args=[rows_to_translate["annotation"]]
        ).get()

    # If we don't need multiprocessing, the code should be as follows
    # dataframe.loc[
    #     rows_to_translate.index, ["annotation", "lang_code", "translation"]
    # ] = translator.transform([rows_to_translate["annotation"]])

    return dataframe


if __name__ == "__main__":
    # Read from SqlAlchemy

    db_connection = create_engine("mysql+pymysql://root:example@localhost:3307")
    df = pd.read_sql(
        "SELECT * FROM newschema.annotation ORDER BY publish_date DESC",
        con=db_connection,
    )

    # If reading from a local file, you can use this instead
    # df = pd.read_csv('./annotation_table.csv', header = 0)

    # We initialise the columns for testing
    df["lang_code"] = None
    df["translation"] = None

    # Divide the comments into chunks for checkpoints
    chunks = np.array_split(range(df.shape[0]), 30)

    # Chunks to ignore in translation (if backups already exist)
    ignore = []

    for i, v in enumerate(chunks):
        if i in ignore:
            pass
        else:
            a = main(df.iloc[v])
            a.to_csv(f"../datasets/translations_{i}.csv", header=True)
