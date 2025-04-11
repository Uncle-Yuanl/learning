#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   build_2048.py
@Time   :   2025/04/08 15:21:59
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用MetaGPT
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

from metagpt.software_company import generate_repo
from metagpt.utils.project_repo import ProjectRepo


repo: ProjectRepo = generate_repo("Create a 2048 game")  # or ProjectRepo("<path>")
"""
创建上下文
ctx = Context()
Team.env = Environment(context=ctx)
	Team.env.context = ctx

雇佣角色
Team.hire(roles)
	Team.env.add_roles
		Team.env.roles[roleprofile] = role
		Team.env.roles[roleprofile].context = Team.env.contenxt
		Team.env.roles[roleprofile].RoleContext.env = Team.env

环境有上下文，上下文也有环境，地址传递
具体角色的环境与公司的环境共享
总体就是设置环境、角色、以及各自之间的上下文关系
"""
print(repo)  # it will print the repo structure with files