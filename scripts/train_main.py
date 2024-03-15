import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd


def direction_model_training():
    def generate_data(sheet_names):
        excel_file_path = 'truncated_normalized__training_dataset.xlsx'
        data_list = []
        labels_list = []

        for idx, sheet_name in enumerate(sheet_names):
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            sensor_data = df.iloc[0:12, :]
            data_list.append(sensor_data.values)
            labels = df.iloc[13, :].values
            labels_list.append(labels)

        # Reshape training_data to [trial_num * [5 * [1 * 12]]]
        reshaped_data_groups = [group.T for group in data_list]
        data = np.stack(reshaped_data_groups, axis=0)
        data = data.astype(np.float32)

        # Reshape training_labels to [trial_num *  [1 * data_length]]
        labels = np.array(labels_list)
        labels = labels.astype(np.int32)

        return data, labels

    def build_lstm_model():
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=20, return_sequences=True,
                       input_shape=(5, 12)))

        # Output layer for multi-class classification (3 classes)
        lstm_model.add(Dense(units=3, activation='softmax'))
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return lstm_model

    def train_model(built_model, training_data, training_labels, validation_data, validation_labels, epochs=400):
        train_history = built_model.fit(training_data, training_labels,
                                        validation_data=(validation_data, validation_labels), epochs=epochs,
                                        batch_size=32)
        return train_history

    def predict(trained_model, data):
        predictions_3class = trained_model.predict(data)
        predictions = np.argmax(predictions_3class, axis=-1)
        max_probabilities = np.max(predictions_3class, axis=-1)
        return predictions, max_probabilities

    def export_results(save_directory):
        training_accuracy_df = pd.DataFrame(history.history['accuracy'])
        training_accuracy_df = training_accuracy_df.T
        training_loss_df = pd.DataFrame(history.history['loss'])
        training_loss_df = training_loss_df.T
        predictions_df = pd.DataFrame(val_predictions)
        max_probabilities_df = pd.DataFrame(val_max_probabilities)
        validation_accuracy_df = pd.DataFrame(history.history['val_accuracy'])
        validation_accuracy_df = validation_accuracy_df.T
        validation_loss_df = pd.DataFrame(history.history['val_loss'])
        validation_loss_df = validation_loss_df.T
        words_df = pd.DataFrame({
            'Val_acc': ['val_acc'],
            'Val_loss': ['val_loss'],
            'Train_acc': ['train_acc'],
            'Train_loss': ['train_loss'],
        })
        words_df = words_df.T

        with pd.ExcelWriter(save_directory + 'validation_predictions.xlsx', engine='xlsxwriter') as writer:
            predictions_df.to_excel(writer, startrow=0, index=False, header=False)
            max_probabilities_df.to_excel(writer, startrow=21, index=False, header=False)
            words_df.to_excel(writer, startrow=44, startcol=0, index=False, header=False)
            validation_accuracy_df.to_excel(writer, startrow=44, startcol=1, index=False, header=False)
            validation_loss_df.to_excel(writer, startrow=45, startcol=1, index=False, header=False)
            training_accuracy_df.to_excel(writer, startrow=46, startcol=1, index=False, header=False)
            training_loss_df.to_excel(writer, startrow=47, startcol=1, index=False, header=False)

    training_sheets = ['011', '012', '013', '014', '015', '021', '022', '023', '024', '025',
                       '031', '032', '033', '034', '035', '041', '042', '043', '044', '045',
                       '051', '052', '053', '054', '055', '061', '062', '063', '064', '065',
                       '071', '072', '073', '074', '075', '081', '082', '083', '084', '085',
                       '091', '092', '093', '094', '095', '101', '102', '103', '104', '105',
                       '111', '112', '113', '114', '115', '121', '122', '123', '124', '125']
    validation_sheets = ['131', '132', '133', '134', '135', '141', '142', '143', '144', '145',
                         '151', '152', '153', '154', '155', '161', '162', '163', '164', '165',]
    train_data, train_labels = generate_data(training_sheets)
    val_data, val_labels = generate_data(validation_sheets)

    # Build the LSTM model
    model = build_lstm_model()

    # Train the model and store the history
    history = train_model(model, train_data, train_labels, val_data, val_labels)
    val_predictions, val_max_probabilities = predict(model, val_data)
    model.save('/4_3-axis_direction_model.h5')

    # Export the results to excel
    save_path = ''
    export_results(save_path)

    criterion = min(history.history['val_loss'])
    return criterion


def depth_model_training():
    def generate_data(sheet_names):
        excel_file_path = 'truncated_normalized_training_dataset.xlsx'
        data_list = []
        labels_list = []

        for idx, sheet_name in enumerate(sheet_names):
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            sensor_data = df.iloc[0:12, :]
            data_list.append(sensor_data.values)
            labels = df.iloc[19, :].values
            labels_list.append(labels)

        # Reshape training_data to [trial_num * [5 * [1 * 12]]]
        reshaped_data_groups = [group.T for group in data_list]
        data = np.stack(reshaped_data_groups, axis=0)
        data = data.astype(np.float32)
        # Reshape training_labels to [trial_num *  [1 * data_length]]
        labels = np.array(labels_list)
        labels = labels.astype(np.int32)

        return data, labels

    def build_lstm_model():
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=20, return_sequences=True,
                       input_shape=(5, 12)))

        # Output layer for multi-class classification (2 classes)
        lstm_model.add(Dense(units=1, activation='sigmoid'))
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return lstm_model

    def train_model(built_model, training_data, training_labels, validation_data, validation_labels, epochs=50):
        train_history = built_model.fit(training_data, training_labels,
                                        validation_data=(validation_data, validation_labels), epochs=epochs,
                                        batch_size=32)
        return train_history

    def predict(trained_model, data):
        predictions_probs = trained_model.predict(data)
        predictions = (predictions_probs > 0.5).astype(int)
        max_probabilities = np.max(predictions_probs, axis=-1)
        return predictions, max_probabilities

    def export_results(save_directory):
        training_accuracy_df = pd.DataFrame(history.history['accuracy'])
        training_accuracy_df = training_accuracy_df.T
        training_loss_df = pd.DataFrame(history.history['loss'])
        training_loss_df = training_loss_df.T
        flattened_predictions = val_predictions.reshape(val_max_probabilities.shape)
        predictions_df = pd.DataFrame(flattened_predictions)
        max_probabilities_df = pd.DataFrame(val_max_probabilities)
        validation_accuracy_df = pd.DataFrame(history.history['val_accuracy'])
        validation_accuracy_df = validation_accuracy_df.T
        validation_loss_df = pd.DataFrame(history.history['val_loss'])
        validation_loss_df = validation_loss_df.T
        words_df = pd.DataFrame({
            'Val_acc': ['val_acc'],
            'Val_loss': ['val_loss'],
            'Train_acc': ['train_acc'],
            'Train_loss': ['train_loss'],
        })
        words_df = words_df.T

        with pd.ExcelWriter(save_directory + 'validation_predictions.xlsx', engine='xlsxwriter') as writer:
            predictions_df.to_excel(writer, startrow=0, index=False, header=False)
            max_probabilities_df.to_excel(writer, startrow=21, index=False, header=False)
            words_df.to_excel(writer, startrow=44, startcol=0, index=False, header=False)
            validation_accuracy_df.to_excel(writer, startrow=44, startcol=1, index=False, header=False)
            validation_loss_df.to_excel(writer, startrow=45, startcol=1, index=False, header=False)
            training_accuracy_df.to_excel(writer, startrow=46, startcol=1, index=False, header=False)
            training_loss_df.to_excel(writer, startrow=47, startcol=1, index=False, header=False)

    training_sheets = ['011', '012', '013', '014', '015', '021', '022', '023', '024', '025',
                       '031', '032', '033', '034', '035', '041', '042', '043', '044', '045',
                       '051', '052', '053', '054', '055', '061', '062', '063', '064', '065',
                       '071', '072', '073', '074', '075', '081', '082', '083', '084', '085',
                       '091', '092', '093', '094', '095', '101', '102', '103', '104', '105',
                       '111', '112', '113', '114', '115', '121', '122', '123', '124', '125']
    validation_sheets = ['131', '132', '133', '134', '135', '141', '142', '143', '144', '145',
                         '151', '152', '153', '154', '155', '161', '162', '163', '164', '165',]
    train_data, train_labels = generate_data(training_sheets)
    val_data, val_labels = generate_data(validation_sheets)

    # Build the LSTM model
    model = build_lstm_model()

    # Train the model and store the history
    history = train_model(model, train_data, train_labels, val_data, val_labels)
    val_predictions, val_max_probabilities = predict(model, val_data)

    # Save the model
    model.save('/Users/Desktop/4_3-axis_depth_model.h5')

    # Export the results to excel
    save_path = ''
    export_results(save_path)

    criterion = min(history.history['val_accuracy'])
    return criterion


def distance_model_training():
    def generate_data(sheet_names):
        excel_file_path = 'normalized_training_dataset.xlsx'
        data_list = []
        labels_list = []

        for idx, sheet_name in enumerate(sheet_names):
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            sensor_data = df.iloc[0:12, 20:]
            labels = df.iloc[21, 20:]

            # sliding window
            sequence_length = sensor_data.shape[1]
            for i in range(sequence_length - 4):
                sequence_data = sensor_data.iloc[:, i:i + 5]
                sequence_labels = labels.iloc[i:i + 5]
                data_list.append(sequence_data.values)
                labels_list.append(sequence_labels.values)

        # Reshape training_data to [trial_num * [5 * [1 * 12]]]
        reshaped_data_groups = [group.T for group in data_list]
        data = np.stack(reshaped_data_groups, axis=0)
        data = data.astype(np.float32)
        # Reshape training_labels to [trial_num *  [1 * data_length]]
        labels = np.array(labels_list)
        labels = labels.astype(np.float32)

        return data, labels

    def build_lstm_model():
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(5, 12)))
        lstm_model.add(LSTM(units=50, return_sequences=True))
        lstm_model.add(Dense(units=1, activation='linear'))  # Changed activation to linear for regression
        lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])  # Changed loss function
        return lstm_model

    def train_model(built_model, training_data, training_labels, validation_data, validation_labels, epochs=400):
        train_history = built_model.fit(training_data, training_labels,
                                        validation_data=(validation_data, validation_labels), epochs=epochs,
                                        batch_size=128)
        return train_history

    def predict(trained_model, data):
        predictions = trained_model.predict(data)
        return predictions

    def export_results(save_directory):
        training_mse_df = pd.DataFrame(history.history['mse'])
        training_mse_df = training_mse_df.T
        training_loss_df = pd.DataFrame(history.history['loss'])
        training_loss_df = training_loss_df.T
        flattened_predictions = np.squeeze(val_predictions, axis=-1)
        predictions_df = pd.DataFrame(flattened_predictions)
        validation_mse_df = pd.DataFrame(history.history['val_mse'])
        validation_mse_df = validation_mse_df.T
        validation_loss_df = pd.DataFrame(history.history['val_loss'])
        validation_loss_df = validation_loss_df.T
        words_df = pd.DataFrame({
            'Val_mse': ['val_mse'],
            'Val_loss': ['val_loss'],
            'Train_mse': ['train_mse'],
            'Train_loss': ['train_loss'],
        })
        words_df = words_df.T

        with pd.ExcelWriter(save_directory + 'validation_predictions.xlsx', engine='xlsxwriter') as writer:
            predictions_df.to_excel(writer, startrow=0, index=False, header=False)
            words_df.to_excel(writer, startrow=22, startcol=0, index=False, header=False)
            validation_mse_df.to_excel(writer, startrow=22, startcol=1, index=False, header=False)
            validation_loss_df.to_excel(writer, startrow=23, startcol=1, index=False, header=False)
            training_mse_df.to_excel(writer, startrow=24, startcol=1, index=False, header=False)
            training_loss_df.to_excel(writer, startrow=25, startcol=1, index=False, header=False)

    training_sheets = ['011', '012', '013', '014', '015', '021', '022', '023', '024', '025',
                       '031', '032', '033', '034', '035', '041', '042', '043', '044', '045',
                       '051', '052', '053', '054', '055', '061', '062', '063', '064', '065',
                       '071', '072', '073', '074', '075', '081', '082', '083', '084', '085',
                       '091', '092', '093', '094', '095', '101', '102', '103', '104', '105',
                       '111', '112', '113', '114', '115', '121', '122', '123', '124', '125']
    validation_sheets = ['131', '132', '133', '134', '135', '141', '142', '143', '144', '145',
                         '151', '152', '153', '154', '155', '161', '162', '163', '164', '165',]
    train_data, train_labels = generate_data(training_sheets)
    val_data, val_labels = generate_data(validation_sheets)

    # Build the LSTM model
    model = build_lstm_model()

    # Train the model and store the history
    history = train_model(model, train_data, train_labels, val_data, val_labels)
    val_predictions = predict(model, val_data)

    # Save the model
    model.save('4_3-axis_distance_model.h5')

    # Export the results to excel
    save_path = ''
    export_results(save_path)

    criterion = min(history.history['val_mse'])
    return criterion

##############################################################################

# Main code
if __name__ == '__main__':
    direction_model_training()
    depth_model_training()
    distance_model_training()
