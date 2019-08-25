import logging
from logging import handlers


filename = 'opinion_mining.log'
fmt = '%(asctime)s - %(funcName)s/%(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s'

logger = logging.getLogger(filename)
format_str = logging.Formatter(fmt)  #设置日志格式
logger.setLevel(logging.DEBUG)  #设置日志级别

sh = logging.StreamHandler()  #屏幕上输出
sh.setFormatter(format_str)  #设置屏幕上显示的格式

th = handlers.TimedRotatingFileHandler(filename=filename, when='D', backupCount=3, encoding='utf-8')
#interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
# S 秒
# M 分
# H 小时、
# D 天、
# W 每星期（interval==0时代表星期一）
# midnight 每天凌晨
th.setFormatter(format_str)

logger.addHandler(sh)
logger.addHandler(th)