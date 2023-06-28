#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   base_demo.py
@Time   :   2023/03/24 16:03:53
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   最基本的一些code
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from flask import Flask, url_for
from flask import request

app = Flask(__name__)

@app.route('/')
def hello_word():
    return "<p>Hello, World!<p>"


# HTML escaping
"""
When returning HTML (the default response type in Flask), 
any user-provided values rendered in the output must be escaped to protect from injection attacks. 
HTML templates rendered with Jinja, introduced later, will do this automatically.
"""
from markupsafe import escape
@app.route('/<name>/')
def hello(name):
    return f"Hello, {escape(name)}"


# URL Building
"""
1. Reversing is often more descriptive than hard-coding the URLs.
2. You can change your URLs in one go instead of needing to remember to manually change hard-coded URLs.
3. URL building handles escaping of special characters transparently.
4. The generated paths are always absolute, avoiding unexpected behavior of relative paths in browsers.
5. If your application is placed outside the URL root, for example, in /myapplication instead of /, url_for() properly handles that for you.
"""
@app.route('/url_build')
def index():
    return 'using url building'


@app.route('/url_build/login')
def login():
    return 'who is using url building'


@app.route('/url_build/user/<username>')
def profile(username):
    return f'{username}\'s profile'


with app.test_request_context():
    # Generate a URL to the given endpoint with the given values.
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/url_build'))
    print(url_for('profile', username='John Doe'))


# HTTP Methods
"""
方法1：
    定义一个函数用来判断并决定处理逻辑，装饰器是路由
    这种方式在每个部分都要使用一些共同数据是很有用
方法2：
    单独使用get装饰器、post装饰器，且视图函数名不同
"""
@app.get('/difhm/')  # also route
def difhm_login_get():
    username = request.args.get('user')
    return f"now the user: {username}"


@app.post('/difhm/')
def difhm_login_post():
    username = request.form['user']
    return f"welcome {username}"


# render template
from flask import render_template
@app.route('/rtemplate/')
@app.route('/rtemplate/<name>')
def remplate(name=None):
    return render_template('hello.html', name=name)