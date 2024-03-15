from keras.models import load_model
import numpy as np
import pandas as pd


def evaluate():
    # direction_model
    def predict(trained_model, data):
        predictions_3class = trained_model.predict(data)
        predictions = np.argmax(predictions_3class, axis=-1)
        max_probabilities = np.max(predictions_3class, axis=-1)
        return predictions, max_probabilities

    # # depth_model
    # def predict(trained_model, data):
    #     predictions_probs = trained_model.predict(data)
    #     predictions = (predictions_probs > 0.5).astype(int)
    #     max_probabilities = np.max(predictions_probs, axis=-1)
    #     return predictions, max_probabilities

    # # distance_model
    # def predict(trained_model, data):
    #     predictions = trained_model.predict(data)
    #     return predictions

    def generate_data(excel_file_path, sheet_name):
        data_list = []
        labels_list = []

        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
        data = df.iloc[0:12, :]  # 4-3axis
        # data = df.iloc[[2, 5, 8, 11], :]  # 4-norm
        # data = df.iloc[0:3, :]  # 1-3axis
        # data = df.iloc[2, :]  # 1-norm
        labels = df.iloc[19, :]

        # sequence_length = data.shape[1]  # for other 3 inputs
        sequence_length = len(data)  # for 1_norm

        # Split data into sequences of length 5 (sliding window approach)
        for i in range(sequence_length - 4):
            # sequence_data = data.iloc[:, i:i + 5]
            # sequence_labels = labels.iloc[i:i + 5]  # for other 3 inputs

            sequence_data = data.iloc[i:i + 5]
            sequence_labels = labels.iloc[i:i + 5]  # for 1-norm

            data_list.append(sequence_data.values)
            labels_list.append(sequence_labels.values)

        reshaped_data_groups = [group.T for group in data_list]
        data = np.stack(reshaped_data_groups, axis=0)
        labels = np.array(labels_list).astype(np.int32)
        data = np.asarray(data).astype(np.float32)  # [(sequence_length-4)*[5*[1*12]]]

        return data, labels

    model_file_path = '4_3-axis_model.h5'
    model = load_model(model_file_path)

    # Define Excel file path and sheet names
    excel_file_path = '/Users/zhengqichen/Desktop/dataset/without winsorization/(0,1)normalized_testing_dataset.xlsx'
    sheet_names = ['171', '172', '181', '182', '191', '192', '201', '202', '211', '212', '221', '222',
                   '231', '232', '241', '242', '251', '252', '261', '262']
    save_directory = '/Users/zhengqichen/Desktop/sliding_window.xlsx'
    start_col = 0  # Starting column index

    with pd.ExcelWriter(save_directory, engine='xlsxwriter') as writer:
        for sheet_name in sheet_names:
            training_data, training_labels = generate_data(excel_file_path, sheet_name)
            predictions, max_probabilities = predict(model, training_data)
            flattened_predictions = predictions.reshape(max_probabilities.shape)
            predictions_df = pd.DataFrame(flattened_predictions)

            results = []
            for predicited_label, actual_label in zip(predictions, training_labels):
                if predicited_label[-1] == actual_label[-1]:
                    results.append(0)
                else:
                    results.append(1)

            last_predictions_df = predictions_df.iloc[:, -1]
            validation_labels_df = pd.DataFrame(training_labels)
            validation_labels_df = validation_labels_df.iloc[:, -1]
            max_probabilities_df = pd.DataFrame(max_probabilities)
            last_probabilities_df = max_probabilities_df.iloc[:, -1]
            results_df = pd.DataFrame(results)
            last_predictions_df.to_excel(writer, startcol=start_col, index=False, header=False)
            validation_labels_df.to_excel(writer, startcol=start_col + 1, index=False, header=False)
            last_probabilities_df.to_excel(writer, startcol=start_col + 2, index=False, header=False)
            results_df.to_excel(writer, startcol=start_col + 3, index=False, header=False)
            start_col += 5

##############################################################################

# Main code
if __name__ == '__main__':
    evaluate()
