#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   solve_func.py
@Time   :   2024/07/09 17:11:47
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   解方程
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from scipy.optimize import fsolve
 
def solve_function(unsolved_value):
    a, b, c, d, e = unsolved_value
    base1 = 59 * a + 32 * b + 8 * c + d
    base2 = 21 * a + 38 * b + 29 * c + 8 * d + 5 * e
    return [
        int((56 * a + 34 * b + 9 * c + d + e) / base1) - 96,
        int((52 * a + 41 * b + 5 * c + d + e) / base1) - 93,
        int((48 * a + 41 * b + 11 * c) / base1) - 88,
        int((18 * a + 39 * b + 31 * c + 7 * d + 6 * e) / base2) - 91,
        int((18 * a + 35 * b + 34 * c + 8 * d + 5 * e) / base2) - 89,
        # 15 * a + 42 * b + 32 * c + 7 * d + 4 * e - base2,
    ]
 
solved=fsolve(solve_function,[1, 1, 1, 1, 1])
print(solved)