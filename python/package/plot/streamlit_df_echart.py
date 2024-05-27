import streamlit as st
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.options import ScatterItem
from pyecharts.commons.utils import JsCode
import base64
import pandas as pd
from pathlib import Path


st.set_page_config(layout='wide')
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
    curdir.parent / "data/Tik Tok Advertisement Review.xlsx",
)
df = df[df["Brand"] != 0]
df = df[df["Image"].str.contains("png")]
df["CTR"] = df["CTR"].str.replace('%', '').astype(int)
df = df.sort_values(by="Brand", ascending=False)

average_x = df["Likes (k)"].mean()
average_y = df["CTR"].mean()

scatter = Scatter(
    init_opts=opts.InitOpts(
        height="1000px"
    )
)
scatter.add_xaxis(df["Likes (k)"].tolist())
num = len(df)
for brand, dfgb in df.groupby(by=["Brand"]):
    for _, row in dfgb.iterrows():
        brandscatteritems = [
            ScatterItem(
                name=row["URL"],
                value=(row["Likes (k)"], row["CTR"])
            )
        ]
        marker = curdir.parent / f"photo/{row['Image']}"
        scatter.add_yaxis(
            series_name=brand[0],
            y_axis=brandscatteritems,
            symbol=f"image://data:image/png;base64,{get_url(marker)}",
            symbol_size=60
        )

scatter.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False),
    legend_opts=opts.LegendOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(is_show=True),
    markline_opts=opts.MarkLineOpts(
        data=[
            opts.MarkLineItem(
                name="Average of Likes",
                x=average_x
            ),
            opts.MarkLineItem(
                name="Average of CTR",
                y=average_y
            )
        ]
    )
)

scatter.set_global_opts(
    legend_opts=opts.LegendOpts(
        type_="scroll",
        pos_left="right",
        pos_bottom=20,
        orient="vertical"
    ),
    xaxis_opts=opts.AxisOpts(
        type_="value",
        name="Likes"
    ),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        name="CTR Top x% ",
        is_inverse=True,
        name_location="start"
    ),
    title_opts=opts.TitleOpts(
        title="Tik Tok Advertisement Review"
    ),
    datazoom_opts=[
        opts.DataZoomOpts(
            is_show=True,
            type_="inside",
            range_start=0,
            range_end=100,
        ),
        opts.DataZoomOpts(
            is_show=True,
            type_="inside",
            range_start=0,
            range_end=100,
            orient="vertical"
        )
    ],
)

events = {
    "click": "function(params) { console.log(params); return [params.name, params.value] }",
    "dblclick": "function(params) { window.open(params.name); return [params.name, params.value] }"
}
# name, value = st_pyecharts(scatter, events=events)
# st.write(name)
# st.write(value)
results = st_pyecharts(scatter, events=events, height="600px")
if results:
    st.write(results)
# st_pyecharts(scatter, events=events)

st.dataframe(df)
# scatter.render("scatter.html")