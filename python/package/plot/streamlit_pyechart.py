from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
import base64

data = [
    [1, 1],
    [2, 4],
    [3, 9],
    # [4, 16]
]

# 加载一个图片作为标记
imgs = []
imgs.append('/home/yhao/code/learning/python/package/plot/markers/D1V1.png')
imgs.append('/home/yhao/code/learning/python/package/plot/markers/C1V1.png')
imgs.append('/home/yhao/code/learning/python/package/plot/markers/C2V1.png')
imgs.append('/home/yhao/code/learning/python/package/plot/markers/C2V1.png')


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img


scatter = Scatter()
scatter.add_xaxis([d[0] for d in data])
num = len(data)
for i in range(num):
    scatter.add_yaxis(
        series_name=i + 1,
        y_axis=[None] * i + [data[i][1]] + [None] * (num - i),
        symbol=f"image://data:image/png;base64,{get_url(imgs[i])}",
        symbol_size=60
        
    )

# scatter.add_yaxis("B", [d[2] for d in data])

scatter.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(is_show=True),
)

scatter.set_global_opts(
    xaxis_opts=opts.AxisOpts(type_="value"),
    yaxis_opts=opts.AxisOpts(type_="value"),
    title_opts=opts.TitleOpts(title="Scatter-Tooltip-Zoom"),
    datazoom_opts=opts.DataZoomOpts(
        is_show=True,
        type_="inside",
        range_start=0,
        range_end=100,
    ),
)

# scatter.render("tooltip_and_zoom.html")

st_pyecharts(scatter)