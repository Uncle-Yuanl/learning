import streamlit as st
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.options import ScatterItem
from pyecharts.commons.utils import JsCode
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
# imgname = "Mock Up Image.png"
imgname = "mock_compress.png"
marker = curdir.parent / f"markers/{imgname}"

scatter = Scatter()
scatter.add_xaxis(df["Likes (k)"].tolist())
num = len(df)
for brand, dfgb in df.groupby(by=["Brand"]):
    brandscatteritems = [
        ScatterItem(
            name="https://www.google.com/",
            value=(row["Likes (k)"], row["CTR"])
        ) for _, row in dfgb.iterrows()
    ]
    scatter.add_yaxis(
        series_name=brand[0],
        y_axis=brandscatteritems,
        # series_name="",
        # y_axis=dfgb["CTR"].tolist(),
        symbol=f"image://data:image/png;base64,{get_url(marker)}",
        symbol_size=60
    )

# 添加超链接
urls = ["https://www.google.com/"] * len(df)
# scatter.add_js_funcs(JsCode(f"window.open('{link}')") for link in enumerate(urls))
chart_id = scatter.chart_id
# js_func = f"""
#     chart_{chart_id}.on('click', function (params) {{
#         window.open({urls[0]}) 
#         console.log(params); 
#     }});
# """
js_func = f"""
    chart_{chart_id}.on('dblclick', function(params) {{
        var opts=option_{chart_id};
        if(params.componentType=="series") {{
            var seriesIndex=params.seriesIndex;
            if (!('markPoint' in opts.series[seriesIndex])) {{
                var markPoint = {{
                    label: {{
                        show: true,
                    }},
                    data: []
                }};
                opts.series[seriesIndex].markPoint = markPoint;
            }}
            var markData={{ name:seriesIndex, coord: params.value, value: params.value[params.value.length-1] }} 
            opts.series[seriesIndex].markPoint.data.push(markData);
            chart_{chart_id}.setOption(opts);
        }} else if(params.componentType=="markPoint") {{
            var seriesIndex=params.seriesIndex;
            var coord=params.data.coord;
            var idxToRemove=opts.series[seriesIndex].markPoint.data.findIndex(function(item) {{
                return item.name == seriesIndex && item.coord[0] === coord[0] && item.coord[1] === coord[1];
            }});
            if (idxToRemove !== -1) {{
                opts.series[seriesIndex].markPoint.data.splice(idxToRemove, 1);
                chart_{chart_id}.setOption(opts);
            }}
        }}
    }});
"""
js_func = [JsCode(f"window.open('{link}')") for link in urls]
scatter.add_js_funcs(*js_func)

scatter.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False),
    legend_opts=opts.LegendOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(is_show=True),
)

scatter.set_global_opts(
    legend_opts=opts.LegendOpts(type_="scroll"),
    xaxis_opts=opts.AxisOpts(type_="value", name="Likes"),
    yaxis_opts=opts.AxisOpts(type_="value", name="CTR"),
    title_opts=opts.TitleOpts(title="Scatter-Tooltip-Zoom"),
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
results = st_pyecharts(scatter, events=events)
if results:
    st.write(results)
# st_pyecharts(scatter, events=events)

# st.dataframe(df)
# scatter.render("scatter.html")