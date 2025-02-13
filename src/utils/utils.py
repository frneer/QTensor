import tensorflow as tf


class VariableHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, variable):
        super(VariableHistoryCallback, self).__init__()
        self.variable = variable
        self.variable_values = []

    def on_epoch_end(self, epoch, logs=None):
        # Record the variable's value at the end of each epoch
        self.variable_values.append(self.variable.numpy())

    def get_history(self):
        return self.variable_values
<<<<<<< Updated upstream
=======

# TODO(Fran): don't do this
def qrange(alpha, signed=True):
    return 2 * alpha if signed else alpha

def delta(alpha, m_levels, signed=True):
    return qrange(alpha, signed) / m_levels

import numpy as np
def plot_snapshot(alpha, levels, thresholds, accuracy, bits, output_path):
    m_levels = 2**bits
    for epoch in range(len(alpha)):

        #pot_alpha = tf.math.round(tf.math.log(alpha[epoch]) / tf.math.log(2.0)) 

        plt.figure()
        # LEVELS
        plt.step(thresholds[epoch], np.concatenate((levels[epoch], [levels[epoch][-1]])), where='post')
        # QUANTIZED LEVELS
        qlevels = delta(alpha[epoch], m_levels) * tf.math.floor(levels[epoch] / delta(alpha[epoch], m_levels))
        plt.step(thresholds[epoch], np.concatenate((qlevels, [qlevels[-1]])), label=f"Quantized levels", linestyle='--', where='post')

        # ALPHA SQUARE
        plt.axvline(x=-alpha[epoch], color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=-alpha[epoch], color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=alpha[epoch], color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=alpha[epoch], color='red', linestyle='--', alpha=0.5)

        plt.text(0.05, 0.95, f"Epoch {epoch + 1}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f"Accuracy: {accuracy[epoch]:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f"Alpha: {alpha[epoch]:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        #plt.text(0.05, 0.75, f"PotAlpha: {pot_alpha:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.legend(loc='lower right')
        max_value = (m_levels - 2) * alpha[epoch] / m_levels

        plt.yticks(np.linspace(levels[epoch][0], max_value, m_levels), labels=[f"" for _ in range(m_levels)])
        plt.grid(which='both', alpha=0.3, linestyle=':')
        plt.savefig(f"{output_path}/snapshot_{epoch:05d}.png")
        plt.close()

        # ffmpeg -framerate 2 -i snapshots/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p snapshots/output.mp4 -y
>>>>>>> Stashed changes
