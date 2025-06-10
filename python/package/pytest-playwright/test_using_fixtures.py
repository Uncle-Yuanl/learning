#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   test_using_fixtures.py
@Time   :   2025/06/10 13:50:47
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习使用fixture，在多个test函数中共享对象
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="function", autouse=True)
def before_each_after_each(page: Page):
    """
    每个test函数都会运行一次
    相当于
    with before_each_after_each() as page:
        test_func(page)
    """
    print("before the test runs")

    # Go to the starting url before each test.
    page.goto("https://playwright.dev/")
    yield
    
    print("after the test runs")


def test_main_navigation_1(page: Page):
    # Assertions use the expect API.
    print("before the test assert 1")
    expect(page).to_have_url("https://playwright.dev/")
    print("after the test assert 1")


def test_main_navigation_2(page: Page):
    # Assertions use the expect API.
    print("before the test assert 2")
    expect(page).to_have_url("https://playwright.dev/")
    print("after the test assert 2")