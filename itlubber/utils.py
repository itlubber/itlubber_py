import os
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, ERROR


font = FontProperties(fname=os.path.join(os.path.dirname(__file__), "YunShuFaJiaYangYongZhiShouJinZhengKaiJian.ttf"))
fp = "fontproperties=font, fontsize=20"


def timer(func):
    """
    function cost time
    """
    def func_wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        print('function {0}() cost time {1} s'.format(func.__name__, time_spend))
        return result

    return func_wrapper


def reduce_memory_usage(df, deep=True, verbose=True):
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = df.memory_usage(deep=deep).sum() / 1024**2

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name

        if verbose and best_type is not None and best_type != str(col_type):
            print(f"column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = df.memory_usage(deep=deep).sum() / 1024**2
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"memory usage decreased from {start_mem:.2f}MB to {end_mem:.2f}MB ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    
    return df


def get_logger(log_format="[ %(levelname)s ][ %(asctime)s ][ %(filename)s:%(funcName)s:%(lineno)d ] %(message)s", filename=None, stream=True):
    logger = getLogger("lubber")
    logger.setLevel(DEBUG)
    formatter = Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    if filename:
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except Exception as error:
                logger.critical(f'错误 >> 创建日志目录失败,清手动创建目录文件位置,运行 sudo mkdir -p {os.path.dirname(filename)}')
                logger.critical('错误 >> 报错信息 : {}'.format(error))

        fh = TimedRotatingFileHandler(filename=filename, when='D', backupCount=30, encoding="utf-8")
        fh.setLevel(INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        fh.close()
    
    if stream:
        ch = StreamHandler()
        ch.setLevel(INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        ch.close()

    return logger


def missing_data(data):
    missing = data.isnull().sum()
    available = data.count()
    total = (missing + available)
    percent = (data.isnull().sum()/data.isnull().count()*100).round(4)
    return pd.concat([missing, available, total, percent], axis=1, keys=['Missing', 'Available', 'Total', 'Percent']).sort_values(['Missing'], ascending=False)


def to_category(df):
    cols = df.select_dtypes(include='object').columns
    for col in cols:
        ratio = len(df[col].value_counts()) / len(df)
        if ratio < 0.05:
            df[col] = df[col].astype('category')
    return df


def date_add(start: str, day: int=None, month: int=None, year: int=None, format: str=None):
    if day:
        if format is None:
            format = "%Y-%m-%d"
        return (datetime.datetime.strptime(start, format) + datetime.timedelta(days=day)).strftime(format)
    if month:
        if format is None:
            format = "%Y-%m"
        return (datetime.datetime.strptime(start, format) + datetime.timedelta(month=month)).strftime(format)
    if year:
        if format is None:
            format = "%Y"
        curr = datetime.datetime.strptime(start, format)
        return curr.strftime(format)
    

if __name__ == '__main__':
    print(date_add("2021-01-12", year=3, format="%Y-%m-%d"))
    