import numpy as np
from flask import Flask, request

#определение необходимых функций для работы
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def predict(x,y):
    sum_t1=W1[0]*x+W1[1]*y+b1
    h=sigmoid(sum_t1)
    sum_t2=h[0][0]*W2[0]+h[0][1]*W2[1]+b2
    o1=sigmoid(sum_t2[0])
    return o1


#добавление правильно обученных весов
W1=np.array([[ 0.08669151, 0.70972208],
                [-0.15009617, 0.33882975]])
b1=np.array([[-2.21261862, 0.0500741 ]])
W2=np.array([[-0.67236509, -0.28416692],
                [-0.18318363, -0.80438505]])
b2=np.array([[0.24321996, 0.62306239]])


#средние значения
sred_r=169.3
sred_v=62.8


app = Flask(__name__)
@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        language = request.form.get('language')
        framework = request.form.get('framework')

        x = np.array([int(language)])
        y = np.array([int(framework)])

        # усреднение данных
        x = x - sred_r
        y = y - sred_v

        rez = predict(x, y)
        if (rez[0] > rez[1]):
            return '''
                  <h1>Это мужчина</h1>'''
        else:
            return '''
                    <h1>Это девушка</h1>'''

    return '''
           <form method="POST">
               <div><label>Language: <input type="text" name="language"></label></div>
               <div><label>Framework: <input type="text" name="framework"></label></div>
               <input type="submit" value="Submit">
           </form>'''
if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)