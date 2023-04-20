from flask import Flask, request, jsonify, send_file
from service.Generator import create
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, flask</p>"


@app.post("/submit")
def create_code():
    data = request.get_json()  # 获取 POST 请求中的 JSON 数据
    res = create(data)
    # 在这里对数据进行处理和操作
    response_data = {"code": res}
    return jsonify(response_data), 200  # 返回 JSON 格式的响应和状态码

@app.get('/download')
def download():
    file_path = './output/cnn'
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=8081)
