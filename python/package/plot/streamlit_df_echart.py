import streamlit as st
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
import base64
import pandas as pd
from pathlib import Path

curdir = Path(__file__)


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img


# 定义过滤函数
def filter_rows(row):
    # 如果第三列的值等于'特定值',返回True跳过该行
    if row[1] == 0:
        return True
    # 否则返回False保留该行
    else:
        return False
    
# 读取数据
df = pd.read_excel(
    curdir.parent / "data/Ad_Clean.xlsx",
    usecols=["Brand", "Likes (k)", "CTR"],
    nrows=200
)
df = df[df["Brand"] != 0]
df["CTR"] = df["CTR"].str.replace('%', '').astype(int)
df["CTR"] = 100 - df["CTR"]

# 读取marker
marker = curdir.parent / "markers/Mock Up Image.png"

scatter = Scatter()
scatter.add_xaxis(df["Likes (k)"].tolist())
num = len(df)
for idx, row in df.iterrows():
    scatter.add_yaxis(
        series_name="",#row["Brand"],
        y_axis=[None] * idx + [row["CTR"]] + [None] * (num - idx),
        symbol=f"image://data:image/png;base64,{get_url(marker)}",
        symbol_size=60
    )


scatter.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False),
    legend_opts=opts.LegendOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(is_show=True),
)

scatter.set_global_opts(
    xaxis_opts=opts.AxisOpts(type_="value", name="Likes"),
    yaxis_opts=opts.AxisOpts(type_="value", name="CTR"),
    title_opts=opts.TitleOpts(title="Scatter-Tooltip-Zoom"),
    datazoom_opts=opts.DataZoomOpts(
        is_show=True,
        type_="inside",
        range_start=0,
        range_end=100,
    ),
)

st_pyecharts(scatter)
st.dataframe(df)