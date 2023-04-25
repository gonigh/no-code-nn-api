from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from service.NetGenerator import createNet
from service.MainGenerator import createMain


app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, flask</p>"


@app.post("/submit")
@cross_origin()
def create_code():
    data = request.get_json()  # 获取 POST 请求中的 JSON 数据
    netCode = createNet(data.get('node'), data.get('edge'))
    mainCode = createMain(data.get('loss'), data.get('optimizer'), data.get('hyperParameters'))
    # 在这里对数据进行处理和操作
    response_data = {"net": netCode, "main": mainCode}
    return jsonify(response_data), 200  # 返回 JSON 格式的响应和状态码


@app.get('/download_net')
@cross_origin()
def downloadNet():
    file_path = './output/net.py'
    return send_file(file_path, as_attachment=True)


@app.get('/download_main')
@cross_origin()
def downloadMain():
    file_path = './output/main.py'
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=8081)
