# save this as app.py
import datetime
import torch

from flask import Flask, request, json

# from main01 import predict

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def hello():
    #默认返回内容
    return_dict = {'return_code':'200','return_info':'处理成功','result':None}

    # 判断传入的json数据是否为空
    if len(request.get_data()) == 0:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)

    print("request.get_data:", request.get_data())
    print("request.get_json:", request.get_json())

    x = request.get_json()
    print("--name:", x['name'])




    # x = [0, 30, 35, 23, 17, 29, 31, 27, 34, 19, 27, 31, 29, 36, 36, 20, 35, 9,
    #      38, 23, 10, 36, 37, 37, 8, 29, 38, 30, 7, 36, 34, 12, 27, 22, 25, 32,
    #      33, 9, 28, 23, 26, 11, 36, 6, 25, 1, 2, 2, 2, 2]
    # x = torch.tensor(x)
    # x = torch.unsqueeze(x, 0)
    #
    # print("预测结果---------------------------------")
    # y = predict(x)
    # print("y:", y)
    # print("y[0]:", y[0])
    #
    # # 对参数进行操作
    # return_dict['result'] = "%s今年%s岁:%s" %(name,age,datetime.datetime.now())
    # print(return_dict)
    return json.dumps(return_dict,ensure_ascii=False)

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")

    app.run(debug=True)
