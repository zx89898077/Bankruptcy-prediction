

from flask import Flask, render_template, request, redirect
import os, csv
# from data import SourceData
from BankruptcyPrediction import PredictBankruptcy

app = Flask(__name__)
#路径改
app.config['FILE_UPLOADS'] = "D:\\Iris\\新加坡项目\\Fin\\phrase2\\project\\ironman-master"


# source = SourceData()
class Data:
    def __init__(self, title=None, values=None, xAxis=None, legend=None, unit=None, special=None):
        self.title = title  # 标题
        self.values = values  # 图表数据
        self.xAxis = xAxis  # 图例名称【一般折线，柱形图需要】
        self.legend = legend  # 横坐标数据【一般折线，柱形图需要】
        self.unit = unit  # 单位【可选】
        self.special = special  # 特殊图标保留参数

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('homepage.html')

@app.route('/performance',methods=['GET', 'POST'])
def performance():
    return render_template('performance.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    # v=[ 2.02590836e-02,  6.59132358e-01, -1.48797771e-02,  9.54218220e-01,
    #     -3.37041726e+01 , 2.02590836e-02 , 2.13899843e-02 , 4.02446166e-01,
    #     1.01546503e+00 , 2.44376711e-01 , 2.13899843e-02 , 7.10979814e-02,
    #     2.41617289e-02 , 2.13899843e-02 , 4.97973389e+03 , 7.42467110e-02,
    #     1.55056522e+00 , 2.13899843e-02 , 1.02967534e-02 , 2.94924288e+01,
    #     1.09079077e+00 , 4.07358364e-02 , 9.74701639e-03 , 2.13899843e-02,
    #     2.44376711e-01 , 7.24454259e-02 , 4.63636810e+02 ,-2.34828331e-02,
    #     4.93311763e+00,  3.33519189e-01 , 1.02967534e-02 , 6.88893269e+01,
    #     6.49498273e+00 , 6.37395903e-02 , 4.07358364e-02 , 2.07399853e+00,
    #     6.41132750e-01  ,5.49772445e-01 , 2.01608440e-02 , 3.31928760e-02,
    #     3.19428952e-01 , 2.01608440e-02 , 6.31800749e+01 , 3.36863590e+01,
    #     3.49861367e-01 , 6.21354956e-01 , 3.01741328e+01 , 1.40083219e-02,
    #     6.29586849e-03 , 5.00135377e-01 , 3.53725916e-01 , 1.88731542e-01,
    #     3.49527135e-01 , 8.23196604e-01 ,-1.27800718e+03 , 1.51641780e-02,
    #     9.57488002e-02 , 9.84837406e-01 , 1.65501067e+00 , 3.30527780e+01,
    #     1.09075326e+01 , 6.75938065e+01 , 6.57669517e+00 , 3.11362172e+00]
    #
    # md="Adaboost"
    # print("Result:",PredictBankruptcy(v,md))
    data = []
    if request.method == 'POST':
        model_name = request.form.get('modelname')
        print(model_name)
        if request.files:
            uploaded_file = request.files['filename'] # This line uses the same variable and worked fine
            filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            uploaded_file.save(filepath)
            print(filepath)
            with open(filepath) as file:
                csv_file = csv.reader(file)
                for row in csv_file:
                    print(row)
            for num in row:
                data.append(float(num))
        print(data)

        result = PredictBankruptcy(data,model_name)
        print("Result:",result)
        return render_template('predict.html',data = result)
    # return redirect(request.url)


    return render_template('predict.html')

# @app.route('/line', methods=['GET', 'POST'])
# def get_data():
#     data = Data()
#     accuracy = 0
#     precision = []
#     recall = []
#     precision_avg = 0
#     recall_avg = 0
#     model_name = request.form.get('model_name')
#     if(model_name == 'Adaboost'):
#     # if(True):
#         accuracy = 0.8111818181818181
#         precision = [0.56039768, 0.5734139 ]
#         recall = [0.50090909, 0.47654545]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     elif(model_name == 'NN'):
#         accuracy = 0.5940909090909091
#         precision = [0.55146667, 0.55386047]
#         recall = [0.44790909, 0.29881818]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     elif(model_name == 'knn'):
#         accuracy = 0.8011818181818182
#         precision = [0.58139059, 0.56513912]
#         recall = [0.45181818, 0.52172727]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     elif(model_name == 'Extreme Gradient Boosting'):
#         accuracy = 0.9210909090909091
#         precision = [0.58148472, 0.58843602]
#         recall = [0.55454545, 0.55136364]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     elif(model_name == 'Random Forest'):
#         accuracy = 0.9183636363636364
#         precision = [0.59116541, 0.58538732]
#         recall = [0.54127273, 0.56527273]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     elif(model_name == 'Bagging'):
#         accuracy = 0.9352727272727271
#         precision = [0.58780269,0.59023041]
#         recall = [0.55927273, 0.565     ]
#         precision_avg = 0.5 * (precision[0] + precision[1])
#         recall_avg = 0.5 * (recall[0] + recall[1])
#     data.title = "可视化图像"
#     data.values = [
#         {'name': 'Adaboost', 'values': [accuracy, precision_avg, recall_avg]},
#     ]
#     data.legend = [key['name'] for key in data.values]
#     data.xAxis = ['accuracy', 'precision', 'recall']
#     data.unit = '单位'
#     return render_template('line.html',title=data.title, data=data.values, legend=data.legend, xAxis=data.xAxis, unit=data.unit,
#                             accuracy = accuracy, precision = precision, recall = recall)
#
# def avg(a ,b):
#     return 0.5 * (a+b)
#
#
# @app.route('/bar', methods=['GET', 'POST'])
# def show_data():
#     data1 = Data()
#     data1.title = "6种模型accuracy对比"
#     data1.values = [
#         {'name': 'Accuracy', 'values': [0.8111818181818181, 0.5940909090909091, 0.8011818181818182, 0.9210909090909091, 0.9183636363636364, 0.9352727272727271]},
#     ]
#     data1.legend = [key['name'] for key in data1.values]
#     data1.xAxis = ['Adaboost', 'NN', 'knn', 'Extreme Gradient Boosting', 'Random Forest', 'Bagging']
#     data1.unit = '单位'
#
#     data2 = Data()
#     data2.title = "6种模型precision对比"
#     data2.values = [
#         {'name': 'Precision', 'values': [avg(0.56039768, 0.5734139), avg(0.55146667, 0.55386047), avg(0.58139059, 0.56513912), avg(0.58148472, 0.58843602), avg(0.59116541, 0.58538732), avg(0.58780269,0.59023041)]},
#     ]
#     data2.legend = [key['name'] for key in data2.values]
#     data2.xAxis = ['Adaboost', 'NN', 'knn', 'Extreme Gradient Boosting', 'Random Forest', 'Bagging']
#     data2.unit = '单位'
#
#     data3 = Data()
#     data3.title = "6种模型recall对比"
#     data3.values = [
#         {'name': 'Recall', 'values': [avg(0.50090909, 0.47654545), avg(0.44790909, 0.29881818), avg(0.45181818, 0.52172727), avg(0.55454545, 0.55136364), avg(0.54127273, 0.56527273), avg(0.55927273, 0.565)]},
#     ]
#     data3.legend = [key['name'] for key in data3.values]
#     data3.xAxis = ['Adaboost', 'NN', 'knn', 'Extreme Gradient Boosting', 'Random Forest', 'Bagging']
#     data3.unit = '单位'
#
#     return render_template('line.html',title1=data1.title, data1=data1.values, legend1=data1.legend, xAxis1=data1.xAxis, unit1=data1.unit,
#                            title2=data2.title, data2=data2.values, legend2=data2.legend, xAxis2=data2.xAxis, unit2=data2.unit,
#                            title3=data3.title, data3=data3.values, legend3=data3.legend, xAxis3=data3.xAxis, unit3=data3.unit,
#                            )
#



if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)
