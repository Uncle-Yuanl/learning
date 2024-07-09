from celery.schedules import crontab


beat_schedule = {
    
}

beat_schedule.update({
    'schedule_task_test': {
        'task': 'proj.tasks.add',  # 这里必须绝对项目路径
        'schedule': crontab(),  # every minute
        'args': (16, 24)
    }
})