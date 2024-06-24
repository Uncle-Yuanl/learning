import streamlit as st
from pyecharts.charts import Scatter
from pyecharts import options as opts
from streamlit_extras.stylable_container import stylable_container
import base64


st.video(
    data="/home/yhao/code/learning/python/package/plot/video/Video zonder titel ‐ Gemaakt met Clipchamp.mp4",
    # data="https://drive.google.com/file/d/1J9JSKPyP-wbA6a4iwLe3jQAwPV_EfLWT/view",
    loop=True,
    autoplay=True
)


# scatter = Scatter(
#     init_opts=opts.InitOpts(
#         height="1000px",
#         bg_color="rgba(0,0,0,0)"
#     ),
#     render_opts=opts.RenderOpts(
#         is_embed_js=True
#     )
# )

# main_bg = "/home/yhao/code/learning/python/package/plot/markers/D1V1.png"
# st.markdown(
#         f"""
#          <style>
#          .stApp {{
#              background: url(data:image/png;base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
#              background-size: cover
#          }}
#          </style>
#          """,
#         unsafe_allow_html=True
# )

# def example():
#     with stylable_container(
#         key="green_button",
#         css_styles="""
#             button {
#                 background-color: green;
#                 color: white;
#                 border-radius: 20px;
#             }
#             """,
#     ):
#         st.button("Green button")

#     st.button("Normal button")

#     with stylable_container(
#         key="container_with_border",
#         css_styles="""
#             {
#                 border: 1px solid rgba(49, 51, 63, 0.2);
#                 border-radius: 0.5rem;
#                 padding: calc(1em - 1px)
#             }
#             """,
#     ):
#         st.markdown("This is a container with a border.")


# example()

# html = """
#     <!-- The video -->
#     <video width="320" height="240" autoplay>
#     <source src="/home/yhao/code/learning/python/package/plot/video/Video zonder titel ‐ Gemaakt met Clipchamp.mp4" type="video/mp4">
#     </video>
# """
# st.html(html)

# st.set_page_config(layout="wide")

# video_html = """
#     <style>

#     #myVideo {
#         position: fixed;
#         right: 0;
#         bottom: 0;
#         min-width: 100%; 
#         min-height: 100%;
#     }

#     .content {
#         position: fixed;
#         bottom: 0;
#         background: rgba(0, 0, 0, 0.5);
#         color: #f1f1f1;
#         width: 100%;
#         padding: 20px;
#     }

#     </style>	
#     # <video autoplay muted loop id="myVideo">
#     #     <source src="https://drive.google.com/file/d/1J9JSKPyP-wbA6a4iwLe3jQAwPV_EfLWT/view">
#     #     Your browser does not support HTML5 video.
#     # </video>
# """

# video_html = """
#     <style>

#     .stVideo {
#         position: fixed;
#         right: 40%;
#         top: 350px;
#     }

#     </style>
# """

# st.markdown(video_html, unsafe_allow_html=True)
# st.title('Video page')

# st.markdown("This text is written on top of the background video! 😁")