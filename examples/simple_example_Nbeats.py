import warnings
import pandas as pd
import numpy as np

# from nbeats_keras.model import NBeatsNet as NBeatsKeras
from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

warnings.filterwarnings(action='ignore', message='Setting attributes')


def main():
    # https://keras.io/layers/recurrent/
    time_steps, input_dim, output_steps =  1, 5, 1

    # Definition of the model.
    # NOTE: If you choose the Keras backend with input_dim>1, you have
    # to set the value here too (in the constructor).
    # model_keras = NBeatsKeras(backcast_length=time_steps, forecast_length=output_steps,
    #                           stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
    #                           nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
    #                           hidden_layer_units=64)

    model_pytorch = NBeatsPytorch(backcast_length=time_steps, forecast_length=output_steps,
                                  stack_types=(NBeatsPytorch.SEASONALITY_BLOCK, NBeatsPytorch.TREND_BLOCK),
                                  nb_blocks_per_stack=1, thetas_dim=(4, 8), share_weights_in_stack=True,
                                  hidden_layer_units=128)#128

    # Definition of the objective function and the optimizer.
    # model_keras.compile(loss='mae', optimizer='adam')
    model_pytorch.compile(loss='mae', optimizer='adam')

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    # where f = np.mean.
    # x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # y = np.mean(x, axis=1, keepdims=True)
    #
    # # Split data into training and testing datasets.
    # c = num_samples // 10
    # x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
    # test_size = len(x_test)
    num_rows = 103

    timesteps = 1
    data = pd.read_csv("E:/下载/n-beats-master新/n-beats-master/examples/data/shanghai1_SEIR.csv")
    data2 = pd.read_csv("E:/下载/n-beats-master新/n-beats-master/examples/data/pre_shanghai1.csv")
    # d = pd.DataFrame(data, columns=['date', 'Days', 'I', 'E', 'R', 'err', 'est_beta'])
    d = pd.DataFrame(data, columns=['Time', 'Estimated_Susceptible', 'Estimated_Exposed','Estimated_Infected','Estimated_Resisatnt','est_beta','err'])
    # d = pd.DataFrame(data, columns=['date','province_confirmedCount','E','R','Days'	,'I','est_beta','err'])

    d2 = pd.DataFrame(data2, columns=['Time', 'Susceptible', 'Exposed', 'Infected', 'Resistant', 'est_beta'])
    # d2 = pd.DataFrame(data2, columns=['Time', 'province_confirmedCount'	,'E','R','Days','I','est_beta'])
    # print(d.head())
    print(d, d2)

    # Use <A, B, C> to predict D.
    predictors = d[['Estimated_Infected','Estimated_Exposed','Estimated_Resisatnt','est_beta']]
    # predictors = d[['I', 'E', 'R', 'est_beta']]
    predictors2 = d2[['Infected', 'Exposed', 'Resistant', 'est_beta']]
    # predictors2 = d2[['I', 'E', 'R', 'est_beta']]
    # predictors = np.log(1+predictors)
    # predictors2=np.log(1+predictors2)
    targets = d['err']
    # targets = np.abs(targets)
    # backcast length is timesteps.
    # forecast length is 1.
    predictors = np.array([predictors[i:i + timesteps] for i in range(num_rows - timesteps + 1)])
    targets = np.array([targets[i:i + 1] for i in range(num_rows - timesteps + 1)])[:, :, None]
    predictors2 = np.array([predictors2[i:i + timesteps] for i in range(7 - timesteps + 1)])
    x_train = predictors[:]
    y_train = targets[:]
    x_test = predictors2[:]
    # Train the model.
    # print('Keras training...')
    # model_keras.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=1)
    # print('Pytorch training...')
    model_pytorch.fit(x_train, y_train, epochs=50, batch_size=1)

    # Save the model for later.4

    # model_keras.save('n_beats_model.h5')0
    # model_pytorch.save('n_beats_pytorch.th')
    # Predict on the testing set (forecast).
    # predictions_keras_forecast = model_keras.predict(x_test)
    # print(predictions_keras_forecast)

    predictions_pytorch_forecast = model_pytorch.predict(x_test)
    predictions_pytorch_forecast=predictions_pytorch_forecast.squeeze(-1)
    print(predictions_pytorch_forecast)

    # np.savetxt('D:/covid_my/SEIR_total1/seir+nbeats3.csv', predictions_pytorch_forecast, delimiter=",")



    # np.testing.assert_equal(predictions_keras_forecast.shape, (test_size, model_keras.forecast_length, output_steps))
    # # np.testing.assert_equal(predictions_pytorch_forecast.shape,
    # #                         (test_size, model_pytorch.forecast_length, output_steps))
    #
    # # Predict on the testing set (backcast).
    # predictions_keras_backcast = model_keras.predict(x_test, return_backcast=True)
    # # predictions_pytorch_backcast = model_pytorch.predict(x_test, return_backcast=True)
    # np.testing.assert_equal(predictions_keras_backcast.shape, (test_size, model_keras.backcast_length, output_steps))
    # np.testing.assert_equal(predictions_pytorch_backcast.shape,
    #                         (test_size, model_pytorch.backcast_length, output_steps))

    # Load the model.
    # model_keras_2 = NBeatsKeras.load('n_beats_model.h5')
    # # model_pytorch_2 = NBeatsPytorch.load('n_beats_pytorch.th')
    #
    # np.testing.assert_almost_equal(predictions_keras_forecast, model_keras_2.predict(x_test))
    # np.testing.assert_almost_equal(predictions_pytorch_forecast, model_pytorch_2.predict(x_test))


if __name__ == '__main__':
    main()
