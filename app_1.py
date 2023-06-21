# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 00:58:31 2022

@author: Ali
"""

from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import base64
from threading import Thread

encd = LabelEncoder()

# Loading the model from disk
saved_model_path = "clinic_NN_model.h5"
reloaded_model = tf.keras.models.load_model(saved_model_path, compile=False)

# Loading classes and features lists
encd.classes_ = np.load('classes.npy', allow_pickle=True)

with open('clinic_features.pkl', 'rb') as f:
    feature_list = pickle.load(f)

# Loading training data for lime interpreter
df = pd.read_csv('Clinic_data.csv')
x = df.drop(['prognosis'], axis=1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Creating lime interpreter
interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns,
    class_names=encd.classes_,
    mode='classification'
)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index_2.html')


from io import BytesIO

from io import BytesIO

def generate_lime_visualization(exp, fig_callback):
    print("Local Explanation:", exp.local_exp)
    print("Available Labels:", exp.available_labels())

    predicted_label = exp.available_labels()[0]  # Use the predicted label
    label_exp = exp.local_exp[predicted_label]

    feature_names = [feature_list[idx] for idx, _ in label_exp]
    importances = [importance for _, importance in label_exp]

    # Increase the plot width by setting the figsize parameter
    fig, ax = plt.subplots(figsize=(9, 5))  # Adjust the width (10 inches) as needed
    ax.barh(feature_names, importances)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Local Feature Importance", fontsize=14)
    plt.subplots_adjust(left=0.28)

    # Save the figure to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Reset the file pointer to the beginning of the file

    # Convert the image data to a base64 string
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")

    # Generate the HTML code to display the image
    html_graph = f'<img src="data:image/png;base64,{img_base64}" alt="LIME Plot">'
    fig_callback(html_graph)



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = str(request.form['message'])

        html_graph = None  # Initialize html_graph outside if-else condition

        if len(list(message.split())) > 1:
            l1 = list(message.split())
            l2 = [int(1) for _ in l1]

            sample_dict = {}
            for i in feature_list:
                sample_dict[i] = 0

            for i, j in zip(l1, l2):
                sample_dict[i] = j

            sam_arr = np.array(list(sample_dict.values())).reshape(1, -1)

            prediction = np.argmax([reloaded_model.predict(sam_arr)][0], axis=1)
            my_prediction = encd.inverse_transform(prediction)

            val = my_prediction.tolist()
            output = val[0]

            exp = interpretor.explain_instance(
                data_row=sam_arr[0],
                predict_fn=reloaded_model.predict,
                top_labels=1
            )

            def set_graph(fig):
                nonlocal html_graph
                plt.close(fig)  # Close the figure to release memory
                html_graph = fig
                print(f"html_graph inside set_graph: {html_graph}")

            lime_thread = Thread(target=generate_lime_visualization, args=(exp, set_graph))
            lime_thread.start()

            lime_thread.join()  # Wait for lime_thread to finish before returning the response

        else:
            output = None

        print(f"html_graph before return: {html_graph}")
        return render_template('after.html', data=output, graph=html_graph)


if __name__ == "__main__":
    app.run(debug=True, threaded=False)
