from flask import Flask, request, jsonify
import sys
import os
import json
import numpy as np
import pandas as pd
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler

from src.Pipeline.predict_pipeline import CustomData
from src.Pipeline.predict_pipeline import Pipeline

application = Flask(__name__)

app=application

#Routes
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        classes = ['Normal', 'Bipolar Type-1', 'Bipolar Type-2', 'Depression']

        requested_data = eval(request.data)

        data = CustomData(
            sadness=requested_data['sadness'],
            euphoric=requested_data['euphoric'],
            exhausted=requested_data['exhausted'],
            sleep_dissorder=requested_data['sleep_dissorder'],
            mood_swing=requested_data['mood_swing'],
            suicidal_thoughts=requested_data['suicidal_thoughts'],
            anorxia=requested_data['anorxia'],
            authority_respect=requested_data['authority_respect'],
            try_explanation=requested_data['try_explanation'],
            aggressive_response=requested_data['aggressive_response'],
            ignore_and_move_on=requested_data['ignore_and_move_on'],
            nervous_break_down=requested_data['nervous_break_down'],
            admit_mistakes=requested_data['admit_mistakes'],
            overthinking=requested_data['overthinking'],
            sexual_activity=requested_data['sexual_activity'],
            concentration=requested_data['concentration'],
            optimisim=requested_data['optimisim']
        )

        pred_df= data.get_data_as_df()
        
        pred_pipeline = Pipeline()
        return jsonify({"diagnosis": classes[pred_pipeline.predict(pred_df)[0]]})

    except:
      return jsonify({"error": "Application Error"})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=os.environ.get('PORT'))  