import webbrowser

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap


# 0:googleMap, 1: 高德地图，2:腾讯地图
def getMapObject(baseSource=1, centerLoc=[0, 0], baseLayerTitle='baseLayer'):
    if baseSource == 0:
        m = folium.Map(location=centerLoc,
                       min_zoom=0,
                       max_zoom=19,
                       zoom_start=8,
                       control=False,
                       control_scale=True
                       )
    elif baseSource == 1:
        # 下面的程式将使用高德地图作为绘图的基底
        m = folium.Map(location=centerLoc,
                       zoom_start=8,
                       control_scale=True,
                       control=False,
                       tiles=None
                       )
        folium.TileLayer(tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                         attr="&copy; <a href=http://ditu.amap.com/>高德地图</a>",
                         min_zoom=0,
                         max_zoom=19,
                         control=True,
                         show=True,
                         overlay=False,
                         name=baseLayerTitle
                         ).add_to(m)
    else:
        # 下面的程式将使用腾讯地图作为绘图的基底
        m = folium.Map(location=centerLoc,
                       zoom_start=8,
                       control_scale=True,
                       control=False,
                       tiles=None
                       )
        folium.TileLayer(tiles='http://rt{s}.map.gtimg.com/realtimerender?z={z}&x={x}&y={y}&type=vector&style=0',
                         attr="&copy; <a href=http://map.qq.com/>腾讯地图</a>",
                         min_zoom=0,
                         max_zoom=19,
                         control=True,
                         show=True,
                         overlay=False,
                         name=baseLayerTitle
                         ).add_to(m)
    return m


# 读取地点构建坐标对
# [[[A.x, A.y], [B.x, B.y]] ... ]
fi = open("\\Data\\20150413\\00005.txt", "r")
COOR = []
pre_lng = 0
pre_lat = 0
while True:
    line = fi.readline()
    if not line:
        break
    else:
        line = line.strip("\n")
        line = line.split(",")
        # 获取经纬度
        lng = float(line[2]) + 0.003162  # 经度
        lat = float(line[3]) - 0.002186  # 维度
        # 画线
        if pre_lng != 0 and pre_lat != 0:
            COOR.append([[pre_lng, pre_lat], [lng, lat]])
        pre_lng, pre_lat = lng, lat
        # 画热力图
        # COOR.append([lat, lng])

Center = [31.220800, 121.466600]

# 地图
m = getMapObject(1, Center, "stamentoner")
# m = folium.Map(location=Center, zoom_start=8.5, control_scale=True)
# 创建geojson图层
# MultiPoint、MultiLineString
gj = folium.GeoJson(data={
    "type": "MultiLineString",
    "coordinates": COOR
})

gj.add_to(m)


# HeatMap(COOR).add_to(m)


# 保存格式为html文件，可使用绝对路径进行保存
name = "Analysis\\traj\\single_traj.html"
m.save(name)

# 将结果文件打开进行显示
webbrowser.open(name, new=2)
