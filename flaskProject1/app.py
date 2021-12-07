from flask import Flask, Response, request, render_template
from werkzeug.utils import secure_filename
import os
from shibie import shibie


xx = "100"
app = Flask(__name__)

# 设置图片保存文件夹
UPLOAD_FOLDER = 'photo'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']


# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS


# 上传图片
@app.route("/upload", methods=['POST', "GET"])
def uploads():
    if request.method == 'POST':
        # 获取post过来的文件名称，从name=file参数中获取
        file = request.files['file']
        if file and allowed_file(file.filename):
            print(file.filename)
            # secure_filename方法会去掉文件名中的中文
            file_name = secure_filename(file.filename)
            global xx
            xx = file_name
            # 保存图片
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

            # return "success"
            return render_template('index1.html')

        else:
            return "格式错误，请上传jpg格式文件"
    return render_template('index.html')


# 查看图片信息
@app.route("/info")
def get_frame():
    return render_template('reader.html')


@app.route("/redex.html")
def lookthepicture():
    s = shibie("123")
    with open(r'D:\BaiduNetdiskDownload\IDA\pycharm\PyCharm 2021.2.3\flaskProject1\photo\{}'.format(xx),
              'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/jpg")
        return resp
    # return render_template('redex.html')


def b(s):
    print("123")
    return s


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/info')
def showinfo():
    return render_template('reader.html')


if __name__ == "__main__":
    app.run()
