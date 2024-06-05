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
minio_endpoint = "https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/"
HEIGHT = 800


# 创建container
container = st.container(
    height=HEIGHT,
    border=True
)


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img


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
            height=f"{HEIGHT}px",
            bg_color="rgba(255,255,255,0.3)"
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
                    value=(row["Likes (k)"], row["CTR"]),
                    tooltip_opts=opts.TooltipOpts(
                        formatter=JsCode((
                            "function(params) {"
                                "var value = params.value;"
                                "var x = value[0];"
                                "var y = value[1];"
                                "return 'Likes (k): ' + x + '<br>CTR Top: ' + y + '%';"
                            "}"
                        ))
                    )
                )
            ]
            scatter.add_yaxis(
                series_name=brand[0],
                y_axis=brandscatteritems,
                symbol=f"image://{minio_endpoint}{row['Image']}",
                symbol_size=120
            )

    scatter.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(is_show=True),
        markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(
                    name="Average of Likes",
                    x=average_x,
                    linestyle_opts=opts.LineStyleOpts(
                        width=8
                    )
                ),
                opts.MarkLineItem(
                    name="Average of CTR",
                    y=average_y,
                    linestyle_opts=opts.LineStyleOpts(
                        width=8
                    )
                )
            ]
        )
    )

    scatter.set_global_opts(
        legend_opts=opts.LegendOpts(
            is_show=False,
            type_="scroll",
            pos_left="right",
            pos_bottom=20,
            orient="vertical"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="value",
            name="Number of Likes (k)",
            is_scale=True,
            name_location="center",
            name_gap=30,
            position="bottom",
            axisline_opts=opts.AxisLineOpts(
                is_on_zero=False,
                symbol=['none', 'arrow']
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=20
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_weight="bold",
                font_size=20
            )
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="CTR Top x% ",
            is_inverse=True,
            name_location="center",
            name_gap=60,
            axisline_opts=opts.AxisLineOpts(
                symbol=['none', 'arrow']
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=20
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_weight="bold",
                font_size=20
            )
        ),
        title_opts=opts.TitleOpts(
            title="Tik Tok Advertisement Review",
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(
                font_weight="bold",
                font_size=50
            )
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
        # "dblclick": "function(params) { window.open(params.name); return [params.name, params.value] }"
        # "dblclick": "function(params) { var newWnd = window.open(); newWnd.opener = null; newWnd.location = params.name; return [params.name, params.value] }"
        "dblclick": "function(params) { setTimeout(() => window.open(params.name, '_blank')); return [params.name, params.value] }"
    }
    results = st_pyecharts(scatter, events=events, height=f"{HEIGHT}px")

# 调整chart position在container中的绝对位置
with container:

    # 调整video透明度
    # video_html = """
    #     <style>

    #     video {
    #         position: relative;
    #         left: 10%;
    #         top: 30px;
    #         width: 80%;
    #         height: 670px;
    #         opacity: 0.6;
    #         filter: brightness(0.6);
    #         object-fit: none;
    #         z-index: 1;
    #     }

    #     </style> 

    #     <video loop muted autoplay>
    #         <source src="https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/tiktok_video.mp4" type="video/mp4" >
    #     </video>

    # """

    background_image = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/page_background.jpg");
            background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
            background-position: center;  
            background-repeat: no-repeat;
        }
        </style>
    """
    
    chart_html = """
        <style>

        iframe {
            position: absolute;
            right: 0px;
            top: 0px;
            z-index: 2;
        }

        </style>

    """
    st.markdown(chart_html, unsafe_allow_html=True)
    st.markdown(background_image, unsafe_allow_html=True)