import numpy as np
from flask import Flask, request


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


@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    if request.method == 'POST':
        rost = request.form.get('rost')
        ves = request.form.get('ves')

        x = np.array([int(rost)])
        y = np.array([int(ves)])

        x = x - sred_r
        y = y - sred_v

        rez = predict(x, y)
        if rez[0] > rez[1]:
            return '''<h1>Это мужчина</h1>'''
        else:
            return '''<h1>Это девушка</h1>'''

    return '''
           <form method="POST">
               <div><label>Рост: <input type="number" name="rost" min="100" max="220"></label></div>
               <div><label>Вес: <input type="number" name="ves" min="30" max="150"></label></div>
               <input type="submit" value="Submit">
           </form>'''


if __name__ == '__main__':
    app.run(debug=True, port=5000)
