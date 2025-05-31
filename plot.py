import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies, epochs, save_path='training_metrics.png'):
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Primary axis for loss
    ax2 = ax1.twinx()  # Secondary axis for accuracy

    # Plot loss
    ax1.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    if val_losses and any(v > 0 for v in val_losses):
        ax1.plot(range(1, epochs + 1), val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(range(1, epochs + 1), train_accuracies, 'b--', label='Train Accuracy')
    if val_accuracies and any(v > 0 for v in val_accuracies):
        ax2.plot(range(1, epochs + 1), val_accuracies, 'r--', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('Training and Validation Metrics')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()