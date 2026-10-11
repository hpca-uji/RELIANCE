import pathlib
import re
import pandas as pd
from typing import List, NoReturn, TextIO
from dataclasses import dataclass, fields

@dataclass(frozen=True)
class PredictionRecord:
    dicom_id: int
    disease: str
    result: bool

def _store_to_csv(dataframe: pd.DataFrame, filename: str, mode: str = 'w', header: bool = False) -> None: # NoReturn
    """
    Store input dataframe into a CSV file.

    Args:
        dataframe (pandas.DataFrame): Dataframe to be stored.
        filename (str): Filename (and path) of the output file.
    
    Keyword args:
        mode (str): Overwrite ('w') or append ('a') data to file, if it exists (Default 'w')
        header (bool): Maintain dataframes header when writing to file (Default False)
    """
    dataframe.to_csv(filename, sep='\t', encoding='utf-8', mode=mode, header=header, index=False)
def store_results(results: List[PredictionRecord], filename: pathlib.Path) -> NoReturn:
    # Create DataFrame
    headers = [f.name for f in fields(PredictionRecord)]
    df = pd.DataFrame(results, columns=headers)
    for f in fields(PredictionRecord):
        if f.type.__name__ == "int":
            df[f.name] = pd.to_numeric(df[f.name])

    # Pivot the DataFrame
    df_index = headers[0]
    df_columns = headers[1:-1]
    df_values = headers[-1]
    #df_pivot = pd.pivot_table(df, index=df_index, columns=df_columns, aggfunc=lambda x: 1, fill_value=0)
    df_pivot = pd.pivot_table(df, index=df_index, columns=df_columns, values=df_values, fill_value=0)

    # Reset index
    df_pivot.reset_index(inplace=True, drop=False)

    # Store to csv
    _store_to_csv(df_pivot, filename, mode='w', header=True)
    #_store_to_csv(dataset, filename, mode='a')

def _read_results(fp: TextIO, pattern) -> List[PredictionRecord]:
    contents: str = fp.read()
    matches = pattern.findall(contents)

    # Trim all whitespaces
    return [(dicom_id.strip(), disease.strip(), result.strip()) for (dicom_id, disease, result) in matches]
def _read_results_file(results_file: pathlib.Path) -> List[PredictionRecord]:
    """    
    Reads the file containing all label names the model will return.
    """
    pattern = re.compile(r"\|(?:\s+([A-Za-z0-9]+\s+)+)\|(\s+(?:[A-Za-z0-9\/]+\s+)+)\|(?:\s+([0-1]+\s+)+)\|", re.IGNORECASE)
    results: PredictionRecord = None
    with open(results_file, "r") as fp:
        results = _read_results(fp, pattern)
    return results

def main():
    in_results_name = "predictions"
    in_results_file = pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(), f"results/{in_results_name}.txt")
    results = _read_results_file(in_results_file)

    out_results_name = "predictions"
    out_results_file = pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(), f"results/{out_results_name}.csv")
    store_results(results, out_results_file)
if __name__ == '__main__':
    main()
