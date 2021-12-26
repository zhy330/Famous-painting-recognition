from flask import Flask,Response, request, render_template
# from flask_cors import CORS
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import random
from aa.testcode import test


app = Flask(__name__)  # 实例化，可视为固定格式
xx = "100"
art_name = "lhb"
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
app.config['JSON_AS_ASCII'] = False  # 解决flask接口中文数据编码问题
# 设置图片保存文件夹
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']


# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS


class Girl:
    def __init__(self, name, addr):
        self.name = name
        self.info = '男'

    def __str__(self):
        return self.name


# 设置可跨域范围
CORS(app, supports_credentials=True)


@app.route('/qqq')
def hello():
    s = r'D:\BaiduNetdiskDownload\IDA\pycharm\PyCharm 2021.2.3\se-flask\static\img\01.jpg'
    art_name = test(s)
    # return xx
    return render_template('index.html')


# 展示Flask如何读取服务器本地图片, 并返回图片流给前端显示的例子
def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

# 跳转到html页面显示图片app.route()为跳转路由，类似springboot


# 上传图片
@app.route("/", methods=['POST', "GET"])
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
            img_p = r'D:\BaiduNetdiskDownload\IDA\pycharm\PyCharm 2021.2.3\se-flask\static\img\{}'.format(xx)
            global art_name
            art_name = test(img_p)
            return render_template('showp.html',file_na=xx,art_na=art_name)

        else:
            return "格式错误，请上传jpg格式文件"
    return render_template('index.html')


@app.route('/showp')
def lookthepicture():
    # img_p = r'D:\BaiduNetdiskDownload\IDA\pycharm\PyCharm 2021.2.3\se-flask\
    img_p = r'static\img\{}'.format(xx)
    # with open(img_p,'rb') as f:
    # image = f.read()
    # resp = Response(image, mimetype="image/jpg")
    # return resp
    return render_template('showp.html', file_na=xx,art_na=art_name)


# 查看图片信息
@app.route("/show")
def get_frame():
    return art_name


@app.route('/index')
def hello_world():
    img_path = 'static/img/02.jpg'
    img_stream = return_img_stream(img_path)
    # render_template()函数是flask函数，它从模版文件夹templates中呈现给定的模板上下文。
    return render_template('post.html', img_stream=img_stream)
    # 主函数


if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    app.run()