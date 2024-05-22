import os
from streamlit_echarts import st_pyecharts
import pyecharts.options as opts
from pyecharts.charts import Line
 
x_data = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
y_data = [820, 932, 901, 934, 1290, 1330, 1320]
 
 
line = Line()
line.set_global_opts(
    tooltip_opts=opts.TooltipOpts(is_show=False),
    xaxis_opts=opts.AxisOpts(type_="category"),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
)
line.add_xaxis(xaxis_data=x_data)
line.add_yaxis(
    series_name="",
    y_axis=y_data,
    symbol="emptyCircle",
    is_symbol_show=True,
    label_opts=opts.LabelOpts(is_show=False),
)
chart_id = line.chart_id
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
line.add_js_funcs(js_func)
# line.render("basic_line_chart.html")
 
# os.startfile("basic_line_chart.html")
st_pyecharts(line)