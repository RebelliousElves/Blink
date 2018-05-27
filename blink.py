from flask import Flask, render_template, request
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

app = Flask("blink")



model_dir_path = './models'

config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()
summarizer = Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

@app.route("/", methods = ['POST', 'GET'])
def home():



    if request.method == 'POST':
        result = request.form
        with graph.as_default():
            headline = summarizer.summarize(result["content"])

        return render_template("index.html", result={'result': headline})

    return render_template("index.html", result={})
