from flask import Flask, request, jsonify
import numpy as np
import pickle as pkl

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit_form():
    data = request.json
    # Process the form data
    gender = data.get('gender')
    age = data.get('age')
    openness = data.get('openness')
    neuroticism = data.get('neuroticism')
    conscientiousness = data.get('conscientiousness')
    agreeableness = data.get('agreeableness')
    extraversion = data.get('extraversion')
    int_features=[gender,age,openness,neuroticism,conscientiousness,agreeableness,extraversion]
    # int_features=[int(x) for x in request.json.values]
    final_features=[np.array(int_features)]
    model=pkl.load(open('model.pkl','rb'))
    output=model.predict(final_features)
    # Perform further operations, such as saving to a database
    # ...

    # Return a response
    # output=np.array(output[0])
    
    # output=[output]
    response = "{}".format(output[0])
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
