import streamlit as st
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.options import ScatterItem
from pyecharts.commons.utils import JsCode
from streamlit_extras.stylable_container import stylable_container
import base64
import pandas as pd
from pathlib import Path


st.set_page_config(layout='wide')
curdir = Path(__file__)
minio_endpoint = "https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/"
WIDTH = 1000
HEIGHT = 500
OPACITY = 0.9
MARKERSIZE = 40

DEFAULT_CATE = "Dressing"
DEFAULT_BRAND = "Hellmann's"


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img

@st.cache_data
def read_data():
    df = pd.read_excel(
        curdir.parent / "data/Tik Tok Advertisement Review_Clean list _Update 1.xlsx"
    )
    df = df[df["Brand"] != 0]
    df["Brand"] = df["Brand"].apply(lambda x: str(x).strip())
    df["Filter by Brand"] = df["Filter by Brand"].apply(lambda x: str(x).strip())
    df = df[df["Image"].str.contains("png|jpg")]
    df["CTR"] = df["CTR"].str.replace('%', '').astype(int)

    return df

@st.cache_data
def init_select(df):
    # By default all brand are selected
    for bf in df["Filter by Brand"].unique():
        st.session_state[bf] = True


df = read_data()
init_select(df)
average_x, max_x = df["Likes (k)"].mean(), df["Likes (k)"].max()
average_y, max_y = df["CTR"].mean(), df["CTR"].max()

# Chart Title
st.markdown("<h1 style='text-align: center; color: black;'>Tik Tok Advertisement Review</h1>", unsafe_allow_html=True)

lb, choosebox, yaxis, chart, rb = st.columns([0.02, 0.18, 0.07, 0.73, 0.1])
with choosebox:
    # Filter header
    filter_html = """
        <!DOCTYPE html>
        <html>
        <style>
            .custom-filter-header {
                position: relative;
                left: 35%;
                width: 80px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: black;
            }
        </style>
        <body>
            <p class="custom-filter-header">&emsp;Filter</p>
        </body>
        </html>
    """
    st.markdown(filter_html, unsafe_allow_html=True)

    with stylable_container(
        key="background_color_checkbox",
        css_styles=f"""
            {{
                background-color: rgba(255,255,255,{OPACITY});
            }}
        """
    ):
        with st.container(height=240):
            st.markdown("<ins>**Filter by Category**</ins>", unsafe_allow_html=True)
            cateset = df["Filter by Category"].unique().tolist()
            cate_check: list[bool] = [
                st.checkbox(label=cate) for cate in cateset
            ]

        with st.container(height=300):
            st.markdown("<ins>**Filter by Brand**</ins>", unsafe_allow_html=True)
            cate_chosen = [cate for cate, check in zip(cateset, cate_check) if check]
            if not any(cate_chosen):
               brandset = df 
            else:
                brandset = df[df["Filter by Category"].isin(cate_chosen)]

            brandset = brandset["Filter by Brand"].unique().tolist()

            for brand in brandset: 
                select = st.checkbox(label=brand, value=st.session_state.get(brand, True))
                st.session_state[brand] = select

            brand_chosen = [brand for brand in df["Filter by Brand"].unique() if st.session_state[brand]]

with chart:
    scatter = Scatter(
        init_opts=opts.InitOpts(
            # width="400px",
            height=f"{HEIGHT}px",
            bg_color=f"rgba(255,255,255,{OPACITY})"
        ),
        render_opts=opts.RenderOpts(
            is_embed_js=True
        )
    )
    scatter.add_xaxis(df["Likes (k)"].tolist())
    num = len(df)
    for brand, dfgb in df.groupby(by=["Filter by Brand"]):
        if brand[0] in brand_chosen:
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
                    series_name=row["Brand"],
                    y_axis=brandscatteritems,
                    symbol=f"image://{minio_endpoint}{row['Image']}",
                    symbol_size=MARKERSIZE
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
            ],
            linestyle_opts=opts.LineStyleOpts(
                color="#00FF7F"
            )
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
            is_scale=True,
            position="bottom",
            min_=0,
            max_=max_x,
            axisline_opts=opts.AxisLineOpts(
                is_on_zero=False,
                symbol=['none', 'arrow']
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=20
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_weight="bold",
                font_size=20,
                background_color="black"
            )
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            is_inverse=True,
            min_=0,
            max_=max_y,
            axisline_opts=opts.AxisLineOpts(
                symbol=['none', 'arrow']
            ),
            axislabel_opts=opts.LabelOpts(
                font_size=20
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_weight="bold",
                font_size=20,
                background_color="black"
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
        # "click": "function(params) { console.log(params); return [params.name, params.value] }",
        "click": "function(params) { window.open(params.name); return [params.name, params.value] }"
        # "dblclick": "function(params) { window.open(params.name); return [params.name, params.value] }"
        # "dblclick": "function(params) { var newWnd = window.open(); newWnd.opener = null; newWnd.location = params.name; return [params.name, params.value] }"
        # "dblclick": "function(params) { setTimeout(() => window.open(params.name, '_blank')); return [params.name, params.value] }",
        # "touch": "function(params) { console.log(params.name); return [params.name, params.value] }"
    }

    results = st_pyecharts(scatter, events=events, width=f"{WIDTH}px", height=f"{HEIGHT}px")

    # X axis
    xaxis_html = """
        <!DOCTYPE html>
        <html>
        <style>
            .custom-xaxis {
                position: relative;
                left: 38%;
                width: 190px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: black;
            }
        </style>
        <body>
            <p class="custom-xaxis">&emsp;Number of Likes (k)</p>
        </body>
        </html>
    """
    st.markdown(xaxis_html, unsafe_allow_html=True)

# Y axis
with yaxis:
    yaxis_html = """
        <!DOCTYPE html>
        <html>
        <style>
            .custom-yaxis {
                position: absolute;
                top: 250px;
                left: -12px;
                width: 115px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: black;
            }
        </style>
        <body>
            <p class="custom-yaxis">&ensp;CTR Top x%</p>
        </body>
        </html>
    """
    st.markdown(yaxis_html, unsafe_allow_html=True)


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
st.markdown(background_image, unsafe_allow_html=True)