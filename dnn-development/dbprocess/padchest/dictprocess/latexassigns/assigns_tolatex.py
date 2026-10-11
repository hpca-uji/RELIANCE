import json
import argparse


"""Function that loads arguments"""


def load_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("assigns_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    args = parser.parse_args()
    return args


"""Function that loads json assign file"""


def _load_json_assigns(path) -> dict:
    with open(path) as jfile:
        data = json.load(jfile)
    return data


"""Function that loads txt assign file and converts it into dictionary"""


def _load_txt_assigns(path) -> dict:

    data = dict()
    with open(path) as rfile:
        for line in rfile:
            linesplit = line.strip().split(" : ")
            key = linesplit[0]
            values = linesplit[1].split(", ")
            data[key] = values
    return data


def load_data(path) -> dict:

    if "txt" in path:
        return _load_txt_assigns(path=path)
    elif "json" in path:
        return _load_json_assigns(path=path)
    else:
        raise ("Unsupported format file")


"""Function that inverses a dictionary from son - father to father - son"""


def get_inverse_dict(sfdict: dict, labelnames: list) -> dict:

    # Create empty dictionary
    inverse_dict = {key: [] for key in labelnames}

    # Traverse sfdict
    for key, values in sfdict.items():
        for v in values:
            inverse_dict[v].append(key)

    return inverse_dict


def pretty_print(print_dict: dict):

    for key, values in print_dict.items():
        if len(values) == 0:
            continue
        print("\\noindent\\textbf{" + key + "}", end=": ")
        for v in values[:-1]:
            print(v, end=", ")
        print(f"{values[-1]}.")
        print()


def main():

    # Load data
    args = load_args()
    sfdict = load_data(args.assigns_path)
    labelnames = load_data(args.labelnames_path)
    # Get inverse dict and print
    inverse_dict = get_inverse_dict(sfdict, labelnames)
    pretty_print(inverse_dict)


if __name__ == "__main__":
    main()
