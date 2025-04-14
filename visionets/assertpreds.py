import torch

a1 = torch.load(
    "/home/fsoler/mldataprojects/reliance/trainedmodels/teacherdn169/dn169_preds.pt"
)
a2 = torch.load("try.pt")
eq = torch.equal(a1, a2)
print(eq)
print(a1)
print(a2)
