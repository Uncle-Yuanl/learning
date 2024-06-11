import streamlit as st
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.options import ScatterItem
from pyecharts.commons.utils import JsCode
from streamlit_extras.stylable_container import stylable_container
from streamlit_js_eval import streamlit_js_eval
import base64
import pandas as pd
from pathlib import Path


st.set_page_config(layout='wide')
curdir = Path(__file__)
minio_endpoint = "https://cmiai-innoflex.unilever-china.com/yhaotemp/photo/"
WIDTH = 800
HEIGHT = 600
OPACITY = 0.9
MARKERSIZE = 40

DEFAULT_CATE = "Dressing"
DEFAULT_BRAND = "Hellmann's"

DISCARDS = [
    "Cheetos",
    "7Up"
]


def get_url(localpath):
    with open(localpath, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded_img


@st.cache_data
def read_data():
    df = pd.read_excel(
        curdir.parent / "data/Tik Tok Advertisement Review_Clean list _Update 1.xlsx"
    )
    df["Brand"] = df["Brand"].apply(lambda x: str(x).strip())
    df["Filter by Brand"] = df["Filter by Brand"].apply(lambda x: str(x).strip())
    df = df[df["Brand"] != 0]
    df = df[~df["Filter by Brand"].isin(DISCARDS)]
    df = df[df["Image"].str.contains("png|jpg")]
    df["CTR"] = df["CTR"].str.replace('%', '').astype(int)

    # make brand-category mapping
    mapping = df.groupby("Filter by Brand").agg(
        {
            "Filter by Category": lambda x: x.tolist()
        }
    )["Filter by Category"].to_dict()

    return df, mapping


@st.cache_data
def init_select(df):
    # By default all brand are selected
    st.session_state["Brands"] = df["Filter by Brand"].unique().tolist()
    st.session_state["ALLCATE"] = True
    select()


def deselect():
    if st.session_state["ALLCATE"]:
        st.session_state["Brands"] = df["Filter by Brand"].unique().tolist()
    for bf in st.session_state["Brands"]:
        st.session_state[bf] = False
    # st.session_state["Brands"] = []

def select():
    if st.session_state["ALLCATE"]:
        st.session_state["Brands"] = df["Filter by Brand"].unique().tolist()
    for bf in st.session_state["Brands"]:
        st.session_state[bf] = True


def filter_category():
    cates = []
    for bcn in st.session_state["Brand_chosen"]:
        cates.extend(st.session_state["BTC"].get(bcn, []))

    st.session_state["Cateset"] = list(set(cates))


def rerun():
    st.session_state["Cateset"] = df["Filter by Category"].unique().tolist()
    st.session_state["ALLCATE"] = True


df, mapping = read_data()
init_select(df)
st.session_state["BTC"] = mapping
average_x, max_x = df["Likes (k)"].mean(), df["Likes (k)"].max()
average_y, max_y = df["CTR"].mean(), df["CTR"].max()

# Chart Title
st.markdown("<h1 style='text-align: center; color: black;'>Tik Tok Advertisement Review</h1>", unsafe_allow_html=True)

lb, choosebox, yaxis, chart, rb = st.columns([0.01, 0.18, 0.1, 0.8, 0.01])
with choosebox:
    # Filter header
    filter_html = """
        <!DOCTYPE html>
        <html>
        <style>
            .custom-filter-header {
                position: relative;
                left: 25%;
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
        with st.container(height=260):
            st.markdown("<ins>**Filter by Category**</ins>", unsafe_allow_html=True)
            cate_empty, reset = st.columns([0.1, 0.9])
            with reset:
                reset_ = st.button(
                    "Reset",
                    use_container_width=True
                )
                if reset_:
                    streamlit_js_eval(js_expressions="parent.window.location.reload()")
            cateset = st.session_state.get(
                "Cateset",
                df["Filter by Category"].unique().tolist()
            )
            cate_check: list[bool] = [
                st.checkbox(label=cate) for cate in cateset
            ]
            st.session_state["Cateset"] = cateset

        with st.container(height=300):
            st.markdown("<ins>**Filter by Brand**</ins>", unsafe_allow_html=True)
            with st.container(height=90, border=False):
                brand_empty, cate_brand_choose = st.columns([0.1, 0.9])
                with cate_brand_choose:
                    deselect_brands = st.button(
                        "Deselect all brands",
                        on_click=deselect,
                        use_container_width=True
                    )
                    doselect_brands = st.button(
                        "Select all brands",
                        on_click=select,
                        use_container_width=True
                    )
                    # filter_cate = st.button(
                    #     "Filter category",
                    #     on_click=filter_category,
                    #     use_container_width=True
                    # )
                    st.markdown(
                        """
                        <style>
                        .stButton {
                            height: 40px;
                            zoom: 67%;
                        }

                        .stCheckbox {
                            zoom: 80%;
                        }

                        # [data-testid=stVerticalBlockBorderWrapper] {
                        #     gap: 0.5rem;
                        # }
                        </style>
                    """,
                        unsafe_allow_html=True,
                    )
            
            cateset = st.session_state["Cateset"]
            # st.write(cateset)
            cate_chosen = [cate for cate, check in zip(cateset, cate_check) if check]
            if not any(cate_chosen):
               brandset = df
               st.session_state["ALLCATE"] = True
            else:
                st.session_state["ALLCATE"] = False
                brandset = df[df["Filter by Category"].isin(cate_chosen)]

            brandset = brandset["Filter by Brand"].unique().tolist()
            st.session_state["Brands"] = brandset

            for brand in st.session_state["Brands"]:
                brand_select = st.checkbox(label=brand, value=st.session_state.get(brand, True))
                st.session_state[brand] = brand_select

            brand_chosen = [brand for brand in st.session_state["Brands"] if st.session_state[brand]]
            st.session_state["Brand_chosen"] = brand_chosen
            # st.write(brand_chosen)


with chart:
    scatter = Scatter(
        init_opts=opts.InitOpts(
            width=f"{WIDTH}px",
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
            min_=1,
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
                left: 35%;
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
                left: -8px;
                width: 100px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: black;
            }
        </style>
        <body>
            <p class="custom-yaxis">&ensp;CTR Rank</p>
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
chart_html = """
    <style>

    iframe {
        zoom: 67%
    }

    </style>

"""
# st.markdown(chart_html, unsafe_allow_html=True)
st.markdown(background_image, unsafe_allow_html=True)