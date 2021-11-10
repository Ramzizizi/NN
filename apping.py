import jsonschema
import numpy as np
from flask import Flask, request, jsonify
from jsonschema import validate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def predict(x, y):
    sum_t1 = W1[0] * x + W1[1] * y + b1
    h = sigmoid(sum_t1)
    sum_t2 = h[0][0] * W2[0] + h[0][1] * W2[1] + b2
    o1 = sigmoid(sum_t2[0])
    return o1


W1 = np.array([[0.08669151, 0.70972208],
               [-0.15009617, 0.33882975]])
b1 = np.array([[-2.21261862, 0.0500741]])
W2 = np.array([[-0.67236509, -0.28416692],
               [-0.18318363, -0.80438505]])
b2 = np.array([[0.24321996, 0.62306239]])

sred_r = 169.3
sred_v = 62.8

app = Flask(__name__)

men = {
    "Пол": "мужчина"
}
women = {
    "Пол": "женщина"
}

schema = {
    "type": "object",
    "properties": {
        "rost": {"type": "number", "minimum": 100, "maximum": 220},
        "ves": {"type": "number", "minimum": 30, "maximum": 150}
    },
}


@app.route("/count", methods=["POST"])
def inf():
    global error
    data = request.get_json()

    try:
        validate(instance=data, schema=schema)
    except jsonschema.ValidationError as er:
        if er.validator == "type":
            error = {"Ошибка": "неверный тип данных"}
        elif er.validator == "minimum":
            error = {"Ошибка": "{} меньше {}".format(er.instance, er.validator_value)}
        else:
            error = {"Ошибка": "{} больше {}".format(er.instance, er.validator_value)}
        return jsonify(error)
    rost = data["rost"]
    ves = data["ves"]

    rost = rost - sred_r
    ves = ves - sred_v

    rez = predict(rost, ves)
    if rez[0] > rez[1]:
        return jsonify(men)
    else:
        return jsonify(women)


@app.errorhandler(500)
def internal_server_error(e):
    error = {"Ошибка": "сбой в программе"}
    return error


app.run()
