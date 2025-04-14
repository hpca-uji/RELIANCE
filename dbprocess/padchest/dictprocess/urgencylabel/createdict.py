import json

import torch
from anytree import Node, RenderTree, PreOrderIter, search


def create_hierarchy():
    """Function that creates hierarchy using anytree"""
    tree = Node(
        "Hierarchy",
        children=[
            Node("Mass"),
            Node(
                "Lung",
                children=[
                    Node(
                        "Lung-urgent",
                        children=[
                            Node("Nodules / Multiple nodules"),
                            Node("Pneumonia"),
                            Node("Infiltrates"),
                            Node("Atelectasis"),
                        ],
                    ),
                    Node("COPD / Emphysema"),
                ],
            ),
            Node(
                "MH",
                children=[
                    Node("MH-urgent", children=[Node("Cardiomegaly"), Node("Hila")]),
                    Node("Vascular hilar enlargement"),
                    Node("Aortic elongation"),
                ],
            ),
            Node(
                "PDTW",
                children=[
                    Node(
                        "PDTW-urgent",
                        children=[
                            Node("Pleural effusion"),
                            Node("Pneumothorax"),
                            Node("Costophrenic angle blunting"),
                        ],
                    ),
                    Node(
                        "PDTW-incidental",
                        children=[Node("Vertebral degenerative changes")],
                    ),
                    Node("Hiatal hernia"),
                ],
            ),
            Node(
                "Foreign bodies",
                children=[
                    Node("Electrical devices"),
                    Node("Tube", children=[Node("NSG Tube")]),
                    Node("Catheter"),
                    Node("Surgery"),
                ],
            ),
            Node("Calcification"),
            Node("Pseudonodule"),
            Node("Suboptimal"),
            Node("Normal"),
        ],
    )
    return tree


def create_sfdict(tree):
    """Creates dictionary son - father diseases"""
    # Compute label_list
    labelnames = [node.name for node in PreOrderIter(tree)][
        1:
    ]  # With [1:] Hierarchy root is discarded

    # Compute son - father dictionary of labels
    sfdict = {}
    for label in labelnames:
        # Find node by name
        node = search.find(tree, filter_=lambda node: node.name == label)
        # Get ancestors names except "Hierarchy" thay goes always first
        lst = [n.name for n in node.ancestors][1:]
        # Assign list to key
        sfdict[label] = lst

    return labelnames, sfdict


def create_hierarchy_matrix(labelnames, sfdict):
    """Creates hierarchy for mcloss using son - father dictionary"""
    # Create device and compute size
    device = torch.device("cpu")
    size = len(labelnames)
    # Create matrix
    hierarchy_matrix = torch.zeros([size, size], dtype=torch.uint8, device=device)
    # Fill matrix
    for i, value1 in enumerate(labelnames):
        for j, value2 in enumerate(labelnames):
            if value1 == value2 or value1 in sfdict[value2]:
                hierarchy_matrix[i, j] = 1
    # Save hierarchy matrix
    return hierarchy_matrix


def create_assigns(fpath, sfdict):
    """Creates assignation of disease to labels"""
    # Open file
    f = open(fpath)
    assigns = {}

    for line in f:

        # Split line into key = disease and values = label
        linesplit = line.strip().split(" : ")
        key = linesplit[0]
        values = linesplit[1].split(", ")
        final_values = set(values)
        # Iterate over values and add ascendants
        for i in range(len(values)):
            final_values.update(sfdict[values[i]])
        assigns[key] = list(final_values)

    return assigns


def main():

    # Create hierarchy
    tree = create_hierarchy()

    # Create labelnames list and sfdict
    labelnames, sfdict = create_sfdict(tree)

    # Save lablnames list
    with open("labelnames.json", "w") as json_file:
        json.dump(labelnames, json_file)

    # Create and save hierarchy matrix
    hierarchy_matrix = create_hierarchy_matrix(labelnames, sfdict)
    torch.save(hierarchy_matrix, "hierarchy.pt")

    # Create disease - label file using dfdict
    assigns = create_assigns("assigns.txt", sfdict)
    """for key, values in assigns.items():
        print(f"{key} : {values}")"""

    # Save assigns
    with open("fullassigns.json", "w") as json_file:
        json.dump(assigns, json_file)


if __name__ == "__main__":
    main()
