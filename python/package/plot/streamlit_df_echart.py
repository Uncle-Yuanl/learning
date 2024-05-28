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

# 创建container
container = st.container(
    height=932,
    border=True
)

# with container:        
#     st.video(
#         data="/home/yhao/code/learning/python/package/plot/video/Video zonder titel ‐ Gemaakt met Clipchamp.mp4",
#         loop=True,
#         autoplay=True
#     )


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img


minio_endpoint = "https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/"

# 读取数据
df = pd.read_excel(
    curdir.parent / "data/Tik Tok Advertisement Review.xlsx",
)
df = df[df["Brand"] != 0]
df = df[df["Image"].str.contains("png")]
df["CTR"] = df["CTR"].str.replace('%', '').astype(int)
# df["CTR"] = 100 - df["CTR"]
df = df.sort_values(by="Brand", ascending=False)

average_x = df["Likes (k)"].mean()
average_y = df["CTR"].mean()


with container:
    scatter = Scatter(
        init_opts=opts.InitOpts(
            height="1000px",
            bg_color="rgba(0,0,0,0)"
        ),
        render_opts=opts.RenderOpts(
            is_embed_js=True
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
            scatter.add_yaxis(
                series_name=brand[0],
                y_axis=brandscatteritems,
                symbol=f"image://{minio_endpoint}{row['Image']}",
                symbol_size=240
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
        # graphic_opts=[
        #     opts.GraphicImage(
        #         graphic_item=opts.GraphicItem(
        #             id_="logo", right=0, top=0, z=-10, bounding="raw", origin=[75, 75]
        #         ),
        #         graphic_imagestyle_opts=opts.GraphicImageStyleOpts(
        #             image="https://echarts.apache.org/zh/images/favicon.png",
        #             width=2309,
        #             height=600,
        #             opacity=0.4,
        #         ),
        #     )
        # ],
    )

    events = {
        "click": "function(params) { console.log(params); return [params.name, params.value] }",
        "dblclick": "function(params) { window.open(params.name); return [params.name, params.value] }"
    }
    # name, value = st_pyecharts(scatter, events=events)
    # st.write(name)
    # st.write(value)
    results = st_pyecharts(scatter, events=events, height="932px")

# 调整chart position在container中的绝对位置
with container:

    # 调整video透明度
    video_html = """
        <style>

        .stVideo {
            # position: fixed;
            # right: 40%;
            # top: 100px;
            # width=1000px;
            # height=600px;
            opacity: 0.8;
        }

        </style>	

    """
    
    chart_html = """
        <style>

        iframe {
            position: absolute;
            right: 0px;
            top: -932px;
        }

        </style>

        <video src="url"  muted  autoplay ></video>

    """
    # st.markdown(video_html, unsafe_allow_html=True)
    # st.markdown(chart_html, unsafe_allow_html=True)

if results:
    st.write(results)
st.dataframe(df)