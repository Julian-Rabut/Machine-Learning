import matplotlib.pyplot as plt
from PIL import Image

def show_images(paths, ncols=4, title=None):
    n = len(paths)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(3*ncols, 3*nrows))
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(img)
        ax.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
