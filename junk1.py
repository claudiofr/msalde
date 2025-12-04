import torch
print(torch.cuda.is_available())   # should return True
print(torch.version.cuda)          # shows CUDA version used

from RandomHingeForest import RandomHingeForest
from packaging.tags import sys_tags
for tag in sys_tags():
    # print(tag)
    pass
num_features=8
numTrees = 4
treeDepth = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print("device",device)
forest = RandomHingeForest(in_channels=num_features, out_channels=numTrees, depth=treeDepth).to(device)

# Process a batch of size 32
x = torch.rand([32, num_features]).to(device)
y = forest(x)
# print(y.cpu())
aa = y.mean(dim=1, keepdim=False)
# aa = y.cpu().mean(dim=1, keepdim=False).to(y.device)
print(aa)
print("yea")