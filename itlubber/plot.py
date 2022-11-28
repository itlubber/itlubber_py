def plot_signal():
    return \
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import seaborn as sns


warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
# plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams["figure.figsize"] = (20, 5)
font = FontProperties(fname=os.path.join(os.path.abspath("."), "YunShuFaJiaYangYongZhiShouJinZhengKaiJian.ttf"))
sns.set(font=font.get_name())
sns.set_style('whitegrid',{'font.sans-serif':['Arial']})
color = ["#2639E9", "#F76E6C", "#FE7715"]


x_cols = "信访件数"
fig = plt.figure(figsize=(20, 5), dpi=120)
ax = fig.subplots(1,1)
ax.plot(data.index, data[x_cols], color=color[0])
# ax.xaxis.set_major_locator(ticker.MultipleLocator(base=60))
ax.tick_params(axis='x', labelrotation=30, labelsize=10, colors=color[0], grid_color=color[0])
ax.tick_params(axis='y', labelrotation=90, labelsize=10, colors=color[0], grid_color=color[0])

ax.spines['top'].set_color(color[0])
ax.spines['bottom'].set_color(color[0])
ax.spines['right'].set_color(color[0])
ax.spines['left'].set_color(color[0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.get_xticklabels()[11].set_color(color[1])

plt.xlabel("信访日期", fontproperties=font_blod, fontsize=20, color=color[0])
plt.ylabel("信访件数", fontproperties=font_blod, fontsize=20, color=color[0])
plt.title("信访件数", fontproperties=font_blod, fontsize=20, color=color[0])

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))

plt.show();
"""


def acf():
    return \
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

x = data["信访件数"]
plot_acf(x)
plot_pacf(x)
"""


def auto_eda(tool="pd"):
    if tool == "sv":
        return \
"""
import sweetviz as sv


advert_report = sv.analyze(data)
advert_report.show_html('data_auto_eda.html')
"""
    else:
        return \
"""
from pandas_profiling import ProfileReport


application_profile = ProfileReport(
                                        data, 
                                        title='Pandas Profiling Report for Application Data', 
                                        html={'style': {'full_width': True}}
                                    ) 
application_profile.to_widgets()
"""


"""
# 两列图
fig, ax = plt.subplots(1, 2)
sb.boxplot(x=col_x, y=target, data=data, ax=ax[0]).set_title(f"Distribution of {target} by {cols_x}")
sb.boxplot(x=col_x, y=target, data=data, ax=ax[1]).set_title(f"Distribution of {target} by {cols_x}")

# 两行图
fig, ax = plt.subplots(2, 1)
sb.boxplot(y=col_x, x=target, data=data, ax=ax[0]).set_title(f"Distribution of {target} by {cols_x}")
sb.boxplot(y=col_x, x=target, data=data, ax=ax[1]).set_title(f"Distribution of {target} by {cols_x}")

# 样本标签分布情况
f, ax = plt.subplots(1, 2, figsize=(20, 10))
data[target].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%' ,ax=ax[0] ,shadow=False)
ax[0].set_title('Distribution of target variable')
ax[0].set_ylabel('')
sns.countplot(target, data=data, ax=ax[1])
ax[1].set_title('Count of good VS bad')
plt.show()


from dataprep.eda import create_report
from dataprep.eda import plot, plot_correlation, plot_missing

# 自定义字体
import matplotlib.pyplot as plt
from matplotlib import font_manager

for font in font_manager.fontManager.ttflist:
    if "line" in font.fname.split("/")[-1]:
        print(font.name)

font_manager.fontManager.addfont("./YunShuFaJiaYangYongZhiShouJinZhengKaiJian.ttf")
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 15
"""
