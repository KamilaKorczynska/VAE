import torch
from pathlib import Path
import json
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.utils.Dinov2.image_dataset import ImageDataset
from src.utils.Dinov2.utils import extract_embeddings, show_similar, make_graph, make_groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device)
model.eval()

root = Path(__file__).resolve().parent.parent.parent.parent
image_dir = root / "data" / "cat"

dataset = ImageDataset(image_dir)
small_subset = Subset(dataset, range(100))

loader = DataLoader(dataset, batch_size=64, shuffle=False)

embeddings, path = extract_embeddings(model, loader, device)
embeddings = F.normalize(
    embeddings,
    p=2,
    dim=1
)
similarity = embeddings @ embeddings.T
similarity.fill_diagonal_(-1)

#show_similar(13, similarity, path)
threshold = 0.93

graph = make_graph(similarity, path, threshold)
groups = make_groups(graph)

groups_paths = [
    [Path(path[i]).name for i in group]
    for group in groups
]

with open("../../datasets/groups_cats.json", "w") as file:
    json.dump(groups_paths, file, indent=4)

