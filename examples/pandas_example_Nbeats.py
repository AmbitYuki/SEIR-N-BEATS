import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.utils.vis_utils import plot_model

from nbeats_keras.model import NBeatsNet as NBeatsKeras

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
#   except RuntimeError as e:
#     print(e)


# This is an example linked to this issue: https://github.com/philipperemy/n-beats/issues/60.
# Here the target variable is no longer part of the inputs.
# NOTE: it is also possible to solve this problem with exogenous variables.
# See example/exo_example.py.

def main():
    num_rows =70
    num_columns = 7
    timesteps = 1
    data = pd.read_csv("D:/下载/n-beats-master新/n-beats-master/examples/data/err_SEIR1.csv")
    d = pd.DataFrame(data, columns=['date', 'Days', 'I', 'E', 'R', 'err','est_beta'])
    # print(d.head())
    print(d)

    # Use <A, B, C> to predict D.
    predictors = d[['I', 'E', 'R','est_beta']]
    # predictors = np.log(1+predictors)
    targets = d['err']
    targets = np.abs(targets)

    # backcast length is timesteps.
    # forecast length is 1.
    predictors = np.array([predictors[i:i + timesteps] for i in range(num_rows - timesteps+1)])
    targets = np.array([targets[i:i + 1] for i in range(num_rows - timesteps+1)])[:, :, None]
    x_train = predictors[:63]
    y_train = targets[:63]
    x_test = predictors[63:]
    y_test = targets[63:]
    print(x_train,y_train,x_test,y_test)
    # noinspection PyArgumentEqualDefault
    model_keras = NBeatsKeras(
        input_dim=num_columns - 3,
        output_dim=1,
        forecast_length=1,
        nb_blocks_per_stack=1,
        backcast_length=timesteps
    )
    # plot_model(model_keras, 'pandas.png', show_shapes=True, show_dtype=True)
    model_keras.compile(loss='mae', optimizer='adam')

    model_keras.fit(x_train, y_train, epochs = 1000)


    # num_predictions = len(predictors)
    predictions = model_keras.predict(x_test)
    print(predictions)
    # predictions.to_csv('D:/下载/n-beats-master新/n-beats-master/examples/data/predictions.csv')
    # np.testing.assert_equal(predictions.shape, (num_predictions, 1, 1))
    # d['P'] = [np.nan] * (num_rows - num_predictions) + list(model_keras.predict(predictors).squeeze(axis=(1, 2)))
    # print(d)


if __name__ == '__main__':
    main()
