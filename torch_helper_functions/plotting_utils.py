"""
This contains utility realted to plotting model information.

## Functions:
  `plot_loss_accuracy_curve`:plot model training and evaluation loss and accuracy curves
"""
import matplotlib.pyplot as plt
from typing import Dict
def plot_model_history_curves(model_history: Dict):
    """
    Plot the loss, accuracy, F1 score from the model history.

    ## Parameters:
        `model_history (Dict`): A dictionary containing the model training and validation history.

    ## Returns:
        None

    ## Example:
        plot_model_history_curves(model_history)
    """
    # Creating a subplot
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # Adjust vertical space between subplots
    fig.subplots_adjust(hspace=0.3)
    ax = ax.ravel()
    title_position = (0.5, 1.05)  # Adjust the position as needed
    epochs = range(1,len(model_0_df)+1)
    fig.suptitle(f"{model_history.get('model_name', 'Model')} Training History", fontsize=18,
                 position=title_position, fontweight='bold')

    # Plotting loss curves for training and validation data
    if 'train_loss' in model_history and 'val_loss' in model_history:
        ax[0].plot(epoch,model_history['train_loss'], label='Training Loss')
        ax[0].plot(epoch,model_history['val_loss'], label='Validation Loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

    # Plotting accuracy curves for training and validation data
    if 'train_acc' in model_history and 'val_acc' in model_history:
        ax[1].plot(epoch,model_history['train_acc'], label='Training Accuracy')
        ax[1].plot(epoch,model_history['val_acc'], label='Validation Accuracy')
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

    # Plotting F1 score curves for training and validation data
    if 'train_f1_score' in model_history and 'val_f1_score' in model_history:
        ax[2].plot(epoch,model_history['train_f1_score'], label='Training F1 Score')
        ax[2].plot(epoch,model_history['val_f1_score'], label='Validation F1 Score')
        ax[2].set_title('F1 Score')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('F1 Score')
        ax[2].legend()

    # Hide the fourth subplot
    ax[3].axis('off')

    plt.show()  # Display the plot
