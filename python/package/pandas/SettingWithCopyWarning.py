#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   SettingWithCopyWarning.py
@Time   :   2023/07/11 13:27:39
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   搞明白SettingWithCopyWarning到底啥问题
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import pandas as pd


def direct_assign():
    """创建df, 直接赋值

    没有warning
    """
    df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['x', 'y', 'z'])

    df['a'] = df['a'].apply(lambda x: str(x) + '_new')

    return df


def direct_partly_assign():
    """创建df, 部分直接赋值

    没有warning
    """
    df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['x', 'y', 'z'])

    dfpart = df[:5]
    print(id(dfpart), id(df))  # 深拷贝
    dfpart['a'] = dfpart['a'].apply(lambda x: str(x) + '_new')

    return df


def address_passing(df):
    """实参，形参（实参的地址），返回值，三者的id都是一样的
    """
    print("接收到的参数的地址：", id(df))
    df['a'] = df['a'].apply(lambda x: str(x) + '_new')

    return df


def address_passing_partly(df):

    print("接收到的参数的地址：", id(df))
    df.loc[:, 'a'] = df['a'].apply(lambda x: str(x) + '_new')

    return df


# ============= inplace ===============
def global_inplace_passing():
    print("函数体内的地址：", id(data))
    data['a'] = data['a'].apply(lambda x: str(x) + '_new')



if __name__ == "__main__":
    # # 1、
    # direct_assign()

    # 2、
    # direct_partly_assign()

    # # 3、
    # data = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['x', 'y', 'z'])
    # print("原始地址：", id(data))
    # res = address_passing_partly(data)
    # print("返回的结果的地址：", id(res))

    # 4、
    data = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['x', 'y', 'z'])
    global_inplace_passing()
    print("函数体内的地址：", id(data))
