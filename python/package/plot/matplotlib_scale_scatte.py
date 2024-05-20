import plotly.graph_objects as go
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as mpimg
import streamlit.components.v1 as components
import mpld3

# def encode_image(img):
#     """
#     编码图像为base64字符串
#     """
#     img_pil = Image.fromarray(img)
#     buffered = BytesIO()
#     img_pil.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return "data:image/png;base64," + img_str


# # 读取要使用的PNG图像文件
# with open("/home/yhao/code/learning/python/package/plot/markers/C1V1.png", "rb") as img_file:
#     encoded_img1 = base64.b64encode(img_file.read()).decode()

# with open("/home/yhao/code/learning/python/package/plot/markers/C2V1.png", "rb") as img_file:
#     encoded_img2 = base64.b64encode(img_file.read()).decode()

# with open("/home/yhao/code/learning/python/package/plot/markers/D1V1.png", "rb") as img_file:
#     encoded_img3 = base64.b64encode(img_file.read()).decode()


# # 创建标记定义列表
# markers = [
#     dict(
#         symbol='url(data:image/png;base64,' + encoded_img1 + ')',
#         size=15
#     ),
#     dict(
#         symbol='url(data:image/png;base64,' + encoded_img2 + ')',
#         size=20
#     ),
#     dict(
#         symbol='url(data:image/png;base64,' + encoded_img3 + ')',
#         size=30
#     )
# ]


# # 生成示例数据
# np.random.seed(42)
# x = np.random.rand(3)
# y = np.random.rand(3)

# # # 加载要在每个点上显示的小图像
# # imgs = [np.random.randint(0, 256, size=(30, 30, 3), dtype=np.uint8) for _ in range(3)]


# # # 创建图像数据
# # images = [go.layout.Image(
# #     source=encode_image(img),
# #     xref="x",
# #     yref="y",
# #     x=x[i],
# #     y=y[i],
# #     sizex=0.2,
# #     sizey=0.2,
# #     sizing="stretch",
# #     opacity=0.8,
# #     layer="below"
# # ) for i, img in enumerate(imgs)]

# # # 创建散点图跟踪
# # trace = go.Scatter(
# #     x=x,
# #     y=y,
# #     mode="markers",
# #     marker=dict(
# #         size=10,
# #         color="rgb(0, 0, 0)",
# #         line=dict(
# #             width=2,
# #             color="rgb(0, 0, 0)"
# #         )
# #     )
# # )

# # # 配置布局
# # layout = go.Layout(
# #     xaxis=dict(
# #         range=[min(x) - 0.1, max(x) + 0.1],
# #         constrain="domain"
# #     ),
# #     yaxis=dict(
# #         range=[min(y) - 0.1, max(y) + 0.1],
# #         constrain="domain"
# #     ),
# #     images=images,
# #     dragmode="zoom"
# # )

# trace = go.Scatter(
#     x=x, y=y,
#     mode="markers",
#     marker=dict(markers)
# )
# # 创建图形
# fig = go.Figure(data=trace)#, layout=layout)

# # 显示图形
# # fig.show()
# st.plotly_chart(fig, theme="streamlit", use_container_width=True)

class PanCanvas:
    def __init__(self, ax):
        self.ax = ax
        self.press = None
        self.x0 = None
        self.y0 = None
 
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
 
    def on_press(self, event):
        if event.button == 2:
            self.press = event.xdata, event.ydata
            self.x0 = self.ax.get_xlim()
            self.y0 = self.ax.get_ylim()
 
    def on_release(self, event):
        if self.press is not None:
            self.press = None
            self.ax.figure.canvas.draw()
 
    def on_motion(self, event):
        if self.press is None:
            return
        if event.button == 2:
            x_press, y_press = self.press
            dx = event.xdata - x_press
            dy = event.ydata - y_press
            self.ax.set_xlim(self.x0[0] - dx, self.x0[1] - dx)
            self.ax.set_ylim(self.y0[0] - dy, self.y0[1] - dy)
            self.ax.figure.canvas.draw()

    # 滚轮放缩
    def on_scroll(self, event):
        """定义鼠标滚轮事件处理函数"""
        base_scale = 1.1
        xdata = event.xdata  # 鼠标在x轴上的坐标
        ydata = event.ydata  # 鼠标在y轴上的坐标
        if event.button == 'up':
            # 向上滚动鼠标滚轮
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # 向下滚动鼠标滚轮
            scale_factor = base_scale
        else:
            # 不处理其他滚轮事件
            return
        # 获取当前坐标系的缩放比例
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        x_range = current_xlim[1] - current_xlim[0]
        y_range = current_ylim[1] - current_ylim[0]
        # 计算新的缩放比例
        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor
        # 计算缩放后的坐标范围
        new_xlim = xdata - (xdata - current_xlim[0])*new_x_range/x_range, \
                xdata + (current_xlim[1] - xdata)*new_x_range/x_range
        new_ylim = ydata - (ydata - current_ylim[0])*new_y_range/y_range, \
                ydata + (current_ylim[1] - ydata)*new_y_range/y_range
        # 更新坐标轴范围
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        # 重新绘制图形
        fig.canvas.draw_idle()


# 生成随机数据
x = np.linspace(0, 10, 3)
y = np.sin(x)
# x = np.random.rand(3)
# y = np.random.rand(3)


# 加载一个图片作为标记
imgs = []
imgs.append(mpimg.imread('/home/yhao/code/learning/python/package/plot/markers/D1V1.png'))
imgs.append(mpimg.imread('/home/yhao/code/learning/python/package/plot/markers/C1V1.png'))
imgs.append(plt.imread('/home/yhao/code/learning/python/package/plot/markers/C2V1.png'))

 
# 创建图形和子图
fig, ax = plt.subplots()
 
# 绘制散点图
ax.scatter(x, y)

# 循环绘制散点图,每个点使用图像作为标记
for i in range(len(x)):
    # im = OffsetImage(imgs[i], zoom=0.05)
    # ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', frameon=False)
    img_base64 = base64.b64encode(imgs[i]).decode('utf-8')
    img_html = '<img src="data:image/png;base64,{}" width="20" height="20">'.format(img_base64)
    ax.annotate(img_html, xy=(x[i], y[i]))
    # ax.add_artist(ab)

# 连接鼠标滚轮事件处理函数
# fig.canvas.mpl_connect('scroll_event', on_scroll)
    
# 创建 PanCanvas 对象
# pan_canvas = PanCanvas(ax)

# # 创建FigureCanvasAgg对象
# canvas = FigureCanvasAgg(fig)

# 在Streamlit上渲染静态图像
# st.pyplot(fig)
plt.show()
# fig_html = mpld3.fig_to_html(fig)
# components.html(fig_html, height=600)

# # 定义JavaScript代码用于处理鼠标事件
# zoom_js = """
#     var fig = window.document.getElementsByTagName('canvas')[0];
#     fig.addEventListener('wheel', function(event) {
#         if (event.deltaY < 0) {
#             Streamlit.setComponentValue(event.deltaY)
#         } else {
#             Streamlit.setComponentValue(event.deltaY)
#         }
#     });
# """

# # 创建一个文本输入框来接收JavaScript发送的事件
# zoom_value = st.slider("Zoom", min_value=-100, max_value=100)

# # 根据输入的缩放值更新图像
# if zoom_value:
#     scale = 1 + int(zoom_value) / 1000
#     ax.set_xlim(np.array(ax.get_xlim()) * scale)
#     ax.set_ylim(np.array(ax.get_ylim()) * scale)
#     canvas.draw()
#     st.pyplot(fig)

# # 在Streamlit上注入JavaScript代码
# st.components.v1.html(
#     f"""
#     <script>
#         {zoom_js}
#     </script>
#     """,
#     height=0,
# )