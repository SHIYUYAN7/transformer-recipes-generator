import matplotlib.pyplot as plt


def plot_loss_and_perplexity(losses, perplexities, loss_title='Training Loss', perplexity_title='Training Perplexity'):
    """
    Plot training loss and perplexity as separate plots. Handles different lengths of metrics.

    Args:
    - losses (list or array): List or array containing loss values per epoch.
    - perplexities (list or array): List or array containing perplexity values per epoch.
    - loss_title (str, optional): Title of the loss plot. Defaults to 'Training Loss'.
    - perplexity_title (str, optional): Title of the perplexity plot. Defaults to 'Training Perplexity'.

    Returns:
    - None: This function will display two separate matplotlib plots.
    """
    # Plot for loss
    epochs_loss = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_loss, losses, marker='o', linestyle='-', color='red')
    plt.title(loss_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Plot for perplexity
    epochs_perplexity = range(1, len(perplexities) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_perplexity, perplexities, marker='o', linestyle='-', color='blue')
    plt.title(perplexity_title)
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.show()


def save_loss_and_perplexity_as_png(losses, perplexities, loss_filename='training_loss.png', perplexity_filename='training_perplexity.png', loss_title='Training Loss', perplexity_title='Training Perplexity'):
    """
    Save training loss and perplexity plots as PNG files.

    Args:
    - losses (list or array): List or array containing loss values per epoch.
    - perplexities (list or array): List or array containing perplexity values per epoch.
    - loss_filename (str, optional): Filename for the loss plot PNG file. Defaults to 'training_loss.png'.
    - perplexity_filename (str, optional): Filename for the perplexity plot PNG file. Defaults to 'training_perplexity.png'.
    - loss_title (str, optional): Title of the loss plot. Defaults to 'Training Loss'.
    - perplexity_title (str, optional): Title of the perplexity plot. Defaults to 'Training Perplexity'.

    Returns:
    - None: This function saves two PNG files for loss and perplexity plots.
    """
    # Plot for loss
    epochs_loss = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_loss, losses, marker='o', linestyle='-', color='red')
    plt.title(loss_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(loss_filename)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory

    # Plot for perplexity
    epochs_perplexity = range(1, len(perplexities) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_perplexity, perplexities, marker='o', linestyle='-', color='blue')
    plt.title(perplexity_title)
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(perplexity_filename)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory