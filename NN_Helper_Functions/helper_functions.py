"""
This helper function is created so that we can reuse some of functions 
created for neuarl networks and deep leraning projects.
Some of the function are:
1) plot_loss_accuracy_curve
2) create_wandb_callback
"""
import pandas as pd
import matplotlib.pyplot as plt
import wandb


def plot_loss_accuracy_curve(history):
    '''
    Plots the model's training and validation loss as well as accuracy curves.

    Parameters:
    history (keras.History): A Keras History object containing training history.

    Returns:
    None
    '''

    # Creating a subplot
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.subplots_adjust(right=2)  # Set left padding in the graph
    ax = ax.ravel()

    # Plotting loss curves for training and validation data
    pd.DataFrame({'Training_Loss': history.history['loss'], 'Val_Loss': history.history['val_loss']}).\
        plot(title='Loss', ax=ax[0])  # Plot loss curve

    # Annotating lowest loss for training and validation data
    ax[0].set_xlabel('Epochs', fontdict={'size': 10})
    ax[0].text(len(history.history['loss']) - 1.2, history.history['loss'][-1],
               'Min_Loss:{}'.format(round(history.history['loss'][-1], 2)),
               fontdict={'size': 15, 'color': 'blue', 'style': 'oblique'})
    ax[0].text(len(history.history['val_loss']) - 1.2, history.history['val_loss'][-1],
               'Min_Val_Loss:{}'.format(
                   round(history.history['val_loss'][-1], 2)),
               fontdict={'size': 15, 'color': 'red', 'style': 'oblique'})

    # Plotting accuracy curves for training and validation data
    pd.DataFrame({'Training_Accuracy': history.history['accuracy'], 'Val_Accuracy': history.history['val_accuracy']})\
        .plot(title='Accuracy', ax=ax[1])  # Plot accuracy curve

    # Annotating highest accuracy for training and validation data
    ax[1].set_xlabel('Epochs', fontdict={'size': 10})
    ax[1].text(len(history.history['accuracy']) - 1.2, history.history['accuracy'][-1],
               'Max_Acc: {}'.format(round(history.history['accuracy'][-1], 2)),
               fontdict={'size': 15, 'color': 'blue', 'style': 'oblique'})
    ax[1].text(len(history.history['val_accuracy']) - 1.2, history.history['val_accuracy'][-1],
               'Max_Val_Acc: {}'.format(
                   round(history.history['val_accuracy'][-1], 2)),
               fontdict={'size': 15, 'color': 'red', 'style': 'oblique'})

    plt.show()  # Display the plot


def create_advanced_wandb_callback(project_name, experiment_name, monitor='val_loss', mode='min', save_model=False):
    '''
    Creates and initializes a customized weight and biases Keras callback for experiment tracking.

    Parameters:
    project_name (str): Name of the Wandb project for experiment tracking.
    experiment_name (str): Name of the specific experiment.
    monitor (str): Quantity to monitor. Default is 'val_loss'.
    mode (str): One of {'auto', 'min', 'max'}. The optimization direction. Default is 'min'.
    save_model (bool): If True, saves the model within the logged experiment. Default is False.

    Returns:
    wandb.keras.WandbCallback: Customized Wandb Callback for Keras.
    '''
    wandb.init(project=project_name, name=experiment_name)
    print(
        f"Logging Wandb experiments with project: {project_name}, name: {experiment_name}")

    # Additional configuration for Wandb callback
    wandb_callback = wandb.keras.WandbCallback(
        monitor=monitor,
        mode=mode,
        save_model=save_model
        # Add other parameters as needed (e.g., save_weights_only, log_weights, log_gradients, etc.)
    )

    return wandb_callback
