import matplotlib.pyplot as plt

def display_images(images, labels, rows=3, cols=5):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*2, rows*2))
    for i, ax in enumerate(axes.flat):
        if i <len(images):
            # Show image
            ax.imshow(images[i], cmap='gray')
            # Show corresponding label
            ax.set_title(f"Label: {labels[i]}")
            # Remove ticks from the plot
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide axes if there are no images to display
            ax.axis('off')

    plt.tight_layout()
    plt.show()