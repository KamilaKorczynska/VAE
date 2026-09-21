from pathlib import Path
from collections import deque
import torch
import matplotlib.pyplot as plt
from PIL import Image


@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_embeddings = []
    all_paths = []

    for idx, (images, paths) in enumerate(loader):
        images = images.to(device)

        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_paths.extend(paths)

        if idx % 5 == 0:
            print(f"Batch: {idx}/{len(loader)}")

    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings, all_paths


def show_similar(index, similarity, paths, k=10):
    values, indices = torch.topk(
        similarity[index],
        k=k
    )

    fig, axs = plt.subplots(
        1,
        k + 1,
        figsize=(4 * (k + 1), 4)
    )

    query = Image.open(paths[index]).convert("RGB")

    axs[0].imshow(query)
    axs[0].set_title("Query")
    axs[0].axis("off")

    for ax, score, i in zip(
        axs[1:],
        values,
        indices
    ):
        image = Image.open(paths[i]).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            f"{score.item():.3f}\n"
            f"{Path(paths[i]).name}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def make_graph(similarity, paths, threshold):
    graph = [[] for _ in range(len(paths))]

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if similarity[i,j] >= threshold:
                graph[i].append(j)
                graph[j].append(i)

    return graph

def make_groups(graph):
    visited = [False] * len(graph)
    groups = []

    for i in range(len(graph)):
        if visited[i] is False:
            group = get_group(start=i, graph=graph, visited=visited)
            groups.append(group)

    return groups

def get_group(start, graph, visited):
    queue = deque()
    group = []

    queue.append(start)
    visited[start] = True

    while queue:
        node = queue.popleft()
        group.append(node)

        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return group


