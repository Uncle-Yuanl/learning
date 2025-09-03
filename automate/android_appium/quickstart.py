#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   quickstart.py
@Time   :   2025/09/02 13:51:20
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用appium链接android设备调试
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import time
from functools import wraps
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def init_driver():
    capabilities = dict(
        platformName='Android',
        automationName='uiautomator2',
        deviceName='emulator-5554',
        appPackage='com.android.settings',
        appActivity='.Settings',
        language='en',
        locale='US'
    )
    driver = webdriver.Remote(
        command_executor=appium_server_url,
        options=UiAutomator2Options().load_capabilities(capabilities)
    )

    return driver


def open_setting_battery():
    # el = driver.find_element(by=AppiumBy.XPATH, value='//*[@text="Battery"]')
    el = driver.find_element(
        by=AppiumBy.ANDROID_UIAUTOMATOR,
        value='new UiScrollable(new UiSelector().scrollable(true)).scrollIntoView(new UiSelector().text("Battery"))'
    )
    el.click()

    time.sleep(10)



def safe_click(by, value, timeout=5):
    """
    装饰器：如果元素存在就点击，不存在则跳过
    :param by: 定位方式 (AppiumBy.ID / AppiumBy.XPATH ...)
    :param value: 元素定位值
    :param timeout: 显式等待超时时间
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                element = self.wait.until(
                    EC.visibility_of_element_located((by, value)),
                    message="元素未出现"
                )
                if element:
                    print(f"[safe_click] 元素 {value} 存在，点击")
                    element.click()
            except TimeoutException:
                print(f"[safe_click] 元素 {value} 不存在，跳过")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator



class DouyinCrawler:
    message = "com.ss.android.ugc.aweme:id/ey7"
    notificationdeny = "com.android.permissioncontroller:id/permission_deny_button"
    update = "com.ss.android.ugc.aweme:id/sj"
    shop = "商城, 按钮"

    def __init__(self):
        self.driver = self.open_app()
        self.wait = WebDriverWait(self.driver, 5)

    def open_app(self):
        capabilities = {
            "platformName": "Android",
            # "platformVersion": "16",
            "deviceName": "emulator-5554",
            "appPackage": "com.ss.android.ugc.aweme",
            "appActivity": "com.ss.android.ugc.aweme.main.MainActivity",
            # "noReset": True,         # 不清除app数据
            # "fullReset": False,
            "automationName": "UiAutomator2",
            "newCommandTimeout": 600   # 单位：秒，设置 10 分钟不操作也不会断开
        }
        driver = webdriver.Remote(
            command_executor=appium_server_url,
            options=UiAutomator2Options().load_capabilities(capabilities)
        )

        return driver

    def portal(self):
        try:
            message = self.wait.until(
                EC.visibility_of_element_located((AppiumBy.ID, self.message))
            )
            
            if message:
                print("Message存在")
                message.click()
        except:
            print("Message元素不存在")

        try:
            notificationdeny = self.wait.until(
                EC.visibility_of_element_located((AppiumBy.ID, self.notificationdeny))
            )
            
            if notificationdeny:
                print("Notification存在")
                notificationdeny.click()
        except:
            print("Notification元素不存在")
        
        try:
            update = self.wait.until(
                EC.visibility_of_element_located((AppiumBy.ID, self.update))
            )
            
            if update:
                print("Update存在")
                update.click()
        except:
            print("Update元素不存在")

    def login(self):
        print()
        me = self.driver.find_element(
            AppiumBy.ACCESSIBILITY_ID,
            "我，按钮"
        )
        me.click()

        numberinput = self.driver.find_element(
            AppiumBy.ID,
            'com.ss.android.ugc.aweme:id/tgr'
        )

        policy = self.driver.find_element(
            AppiumBy.ID,
            'com.ss.android.ugc.aweme:id/uuw'
        )
        
        verify = self.driver.find_element(
            AppiumBy.CLASS_NAME,
            'android.widget.Button'
        )



    
    def switch_to_shop(self):
        tt = 0
        while True:
            try:
                shop = self.driver.find_element(
                    AppiumBy.ACCESSIBILITY_ID,
                    "商城，按钮"
                )
            except:
                tt += 1
                time.sleep(2)
            
            else:
                print("Update存在")
                shop.click()
            
            
            if tt > 5:
                print("Shop元素不存在")
                raise ModuleNotFoundError

    def pipeline(self):
        self.portal()
        self.login()
        self.switch_to_shop()

    def get_comments(self):
        pass

    

if __name__ == "__main__":
    appium_server_url = 'http://localhost:4723'

    dc = DouyinCrawler()

    try:
        dc.pipeline()

    except Exception as e:
        print(str(e))

    finally:
        dc.driver.quit()
