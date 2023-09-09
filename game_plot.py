import matplotlib.pyplot as plt
from IPython import display

plt.ion() # it is a command which enables the real time updation of the plot

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf()) # Ipython command, displays the current matplotlib figure
    plt.clf() # it removes the previous lines and labels from the plot, preparing it for the next frame
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Specify colors for the lines
    plt.plot(scores, label='Scores', color='green')
    plt.plot(mean_scores, label='Mean Scores', color='orange')
    
    plt.ylim(ymin=0) # ensures that the minimum value of y-axis is 0
    plt.text(len(scores)-1, scores[-1], str(scores[-1])) # 3 argumnets, x-coordinate, y-coordinate & text to display
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    # legend to the plot
    plt.legend(loc='upper right', labels=['Scores', 'Mean Scores'], bbox_to_anchor=(1, 1))
    
    plt.show(block=False) # displays the plot without blocking the execution
    plt.pause(.2) # adds a small delay for smoother animation
