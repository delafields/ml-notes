import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow import keras
import numpy as np

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @parm fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """

    # draw the renderer
    fig.canvas.draw()

    # get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape(w, h, 4)

    # canvas.tostring_argb gives pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def repeated_predictions(model, data, lookback, steps=100):
    predictions = []
    for i in range(steps):
        input_data = data[np.newaxis, :, np.newaxis]
        generated = model.predict(input_dat)[0]
        data = np.append(data, generated)[-look_back:]
        predictions.append(generated)
    return predictions

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, trainX, trainY, testX, testY, look_back, repeated_predictions=False):
        self.repeated_predictions = repeated_predictions
        self.trainX = trainX
        self.trainY = trainY
        self.testX  = testX
        self.testY  = testY
        self.look_back = look_back

    def on_epoch_end(self, epoch, logs):
        if self.repeated_predictions:
            preds = repeated_predictions(
                self.model, self.trainX[-1, :, 0], self.look_back, self.testX.shape[0]
            )
        else:
            preds = self.model.predict(self.testX)

        # generate a figure with matplotlib</font>
        figure = plt.figure(figsize=(10,10))
        plot = figure.add_subplot(111)

        plot.plot(self.trainY)
        plot.plot(np.append(np.empty_like(self.trainY) * np.nan, self.testY))
        plot.plot(np.append(np.empty_like(self.trainY) * np.nan, preds))

        data = fig2data(figure)
        plt.close(figure)