import geopandas
import folium
from folium.plugins import HeatMap
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
import os
import json # 用于处理GeoJSON字符串等

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "geospatial_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEFAULT_CRS = "EPSG:4326" # 默认地理坐标系 (WGS84)

# ==============================================================================
# GeoDataFrame 创建与读取函数 (GeoDataFrame Creation and Reading Functions)
# ==============================================================================

def create_gdf_from_dict(data_dict, geometry_col='geometry', crs=DEFAULT_CRS):
    """
    从字典创建GeoDataFrame。

    参数:
    data_dict (dict): 包含属性数据和geometry对象的字典。
                      geometry列应包含Shapely Geometry对象。
                      示例: {'city': ['A'], 'geometry': [Point(0,0)]}
    geometry_col (str, optional): 几何列的名称。默认为 'geometry'。
    crs (str or dict, optional): 坐标参考系统。默认为 "EPSG:4326"。

    返回:
    geopandas.GeoDataFrame: 创建的GeoDataFrame。
    """
    try:
        return geopandas.GeoDataFrame(data_dict, geometry=geometry_col, crs=crs)
    except Exception as e:
        print(f"从字典创建GeoDataFrame失败: {e}")
        return geopandas.GeoDataFrame()

def read_gdf_from_file(file_path, crs=None, **kwargs):
    """
    从多种地理空间文件格式读取数据到GeoDataFrame。
    支持常见的格式如 Shapefile (.shp), GeoJSON (.geojson), GeoPackage (.gpkg) 等。

    参数:
    file_path (str): 地理空间文件的路径。
    crs (str or dict, optional): 强制设置的坐标参考系统。如果文件本身包含CRS信息，通常不需要此参数。
    **kwargs: 传递给 geopandas.read_file 的其他参数。

    返回:
    geopandas.GeoDataFrame: 读取的GeoDataFrame。
    """
    try:
        gdf = geopandas.read_file(file_path, **kwargs)
        if crs and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        return gdf
    except Exception as e:
        print(f"从文件 '{file_path}' 读取GeoDataFrame失败: {e}")
        return geopandas.GeoDataFrame()

def create_gdf_from_latlon_df(df, longitude_col, latitude_col, crs=DEFAULT_CRS):
    """
    从包含经纬度列的Pandas DataFrame创建点状GeoDataFrame。

    参数:
    df (pd.DataFrame): 包含经纬度数据的Pandas DataFrame。
    longitude_col (str): 经度列的名称。
    latitude_col (str): 纬度列的名称。
    crs (str or dict, optional): 坐标参考系统。默认为 "EPSG:4326"。

    返回:
    geopandas.GeoDataFrame: 创建的点状GeoDataFrame。
    """
    try:
        geometry = [Point(xy) for xy in zip(df[longitude_col], df[latitude_col])]
        return geopandas.GeoDataFrame(df, geometry=geometry, crs=crs)
    except Exception as e:
        print(f"从经纬度DataFrame创建GeoDataFrame失败: {e}")
        return geopandas.GeoDataFrame()

# ==============================================================================
# GeoPandas 基本操作接口 (GeoPandas Basic Operations Interface)
# ==============================================================================

def get_gdf_info(gdf):
    """打印GeoDataFrame的简明摘要 (类似 .info())。"""
    if isinstance(gdf, geopandas.GeoDataFrame):
        print("\n--- GeoDataFrame Info ---")
        gdf.info()
    else:
        print("输入不是有效的GeoDataFrame。")

def get_gdf_head(gdf, n=5):
    """返回GeoDataFrame的前n行。"""
    if isinstance(gdf, geopandas.GeoDataFrame):
        print(f"\n--- GeoDataFrame Head (first {n} rows) ---")
        print(gdf.head(n))
        return gdf.head(n)
    else:
        print("输入不是有效的GeoDataFrame。")
        return pd.DataFrame()


def get_gdf_crs(gdf):
    """获取并打印GeoDataFrame的坐标参考系统。"""
    if isinstance(gdf, geopandas.GeoDataFrame) and hasattr(gdf, 'crs'):
        print(f"\n--- CRS (Coordinate Reference System) ---")
        print(gdf.crs)
        return gdf.crs
    else:
        print("无法获取CRS：输入不是有效的GeoDataFrame或缺少CRS信息。")
        return None

def reproject_gdf(gdf, target_crs):
    """
    将GeoDataFrame重新投影到目标坐标参考系统。

    参数:
    gdf (geopandas.GeoDataFrame): 输入的GeoDataFrame。
    target_crs (str or dict): 目标CRS (例如 "EPSG:3857" 或 {'init': 'epsg:3857'})。

    返回:
    geopandas.GeoDataFrame: 重新投影后的GeoDataFrame。
    """
    if not isinstance(gdf, geopandas.GeoDataFrame) or not hasattr(gdf, 'crs') or not gdf.crs:
        print("无法重投影：输入不是有效的GeoDataFrame或缺少源CRS。")
        return gdf
    try:
        return gdf.to_crs(target_crs)
    except Exception as e:
        print(f"重投影到 {target_crs} 失败: {e}")
        return gdf

def get_gdf_area(gdf, projected_crs_for_meaningful_area="EPSG:3395"):
    """
    计算GeoDataFrame中每个几何对象的面积。
    如果CRS是地理坐标系 (如EPSG:4326)，面积单位是平方度，可能没有实际意义。
    函数会尝试重投影到指定的投影坐标系 (默认为World Mercator EPSG:3395) 以获得更有意义的面积 (如平方米)。
    
    参数:
    gdf (geopandas.GeoDataFrame): 输入的GeoDataFrame。
    projected_crs_for_meaningful_area (str, optional): 用于计算有意义面积的投影CRS。

    返回:
    pd.Series or None: 包含面积的Series，如果失败则返回None。
    """
    if not isinstance(gdf, geopandas.GeoDataFrame) or not hasattr(gdf, 'geometry'):
        print("输入不是有效的GeoDataFrame或缺少几何列。")
        return None
    
    temp_gdf = gdf
    meaningful_area_calculated = False
    if temp_gdf.crs and temp_gdf.crs.is_geographic:
        print(f"警告: GeoDataFrame的CRS '{temp_gdf.crs}' 是地理坐标系。面积将以平方度为单位。")
        print(f"尝试重投影到 '{projected_crs_for_meaningful_area}' 以计算更有意义的面积。")
        try:
            temp_gdf = temp_gdf.to_crs(projected_crs_for_meaningful_area)
            meaningful_area_calculated = True
            print(f"已重投影到 {projected_crs_for_meaningful_area}。面积单位将是该投影坐标系的平方单位 (通常是米)。")
        except Exception as e:
            print(f"重投影到 {projected_crs_for_meaningful_area} 失败: {e}. 面积仍将基于原CRS计算。")
    elif not temp_gdf.crs:
        print("警告: GeoDataFrame缺少CRS信息。面积计算可能不准确或单位未知。")
    
    try:
        areas = temp_gdf.area
        if meaningful_area_calculated:
            print("面积计算完成 (基于投影坐标系)。")
        return areas
    except Exception as e:
        print(f"计算面积失败: {e}")
        return None

def get_gdf_centroid(gdf):
    """计算GeoDataFrame中每个几何对象的质心。"""
    if not isinstance(gdf, geopandas.GeoDataFrame) or not hasattr(gdf, 'geometry'):
        print("输入不是有效的GeoDataFrame或缺少几何列。")
        return None
    try:
        return gdf.centroid
    except Exception as e:
        print(f"计算质心失败: {e}")
        return None

# ==============================================================================
# GeoPandas 绘图函数 (GeoPandas Plotting Functions)
# ==============================================================================

def plot_geopandas(gdf, column_to_plot=None, cmap='viridis', title='GeoPandas Plot', 
                   legend=True, add_basemap=False, basemap_crs="EPSG:3857", 
                   basemap_source=cx.providers.OpenStreetMap.Mapnik,
                   figsize=(10,10), save_filename_png=None, **plot_kwargs):
    """
    使用 GeoPandas 绘制地理空间数据，可选添加Contextily底图。

    参数:
    gdf (geopandas.GeoDataFrame): 要绘制的GeoDataFrame。
    column_to_plot (str, optional): 用于颜色编码 (choropleth) 的列名。
    cmap (str, optional): Matplotlib颜色映射表。
    title (str, optional): 图表标题。
    legend (bool, optional): 是否显示图例 (如果column_to_plot被指定)。
    add_basemap (bool, optional): 是否添加Contextily底图。
    basemap_crs (str, optional): 底图的CRS，通常是Web Mercator "EPSG:3857"。
    basemap_source (contextily.tile_providers object, optional): Contextily底图源。
    figsize (tuple, optional): Matplotlib图像大小。
    save_filename_png (str, optional): PNG图像的保存名 (例如 'my_plot.png')。如果None，则尝试plt.show()。
    **plot_kwargs: 传递给 gdf.plot() 的其他关键字参数。
    """
    if not isinstance(gdf, geopandas.GeoDataFrame) or gdf.empty:
        print("无法绘图：输入不是有效的GeoDataFrame或GeoDataFrame为空。")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 确定实际用于绘图的列
    plot_col_arg = None
    if column_to_plot and column_to_plot in gdf.columns:
        if gdf[column_to_plot].dtype != 'geometry':
            plot_col_arg = column_to_plot
        else:
            print(f"警告: 指定的列 '{column_to_plot}' 是几何列，不能用于颜色编码。将进行默认绘图。")
    elif column_to_plot:
        print(f"警告: 指定的列 '{column_to_plot}' 不在GeoDataFrame中。将进行默认绘图。")

    gdf.plot(ax=ax, column=plot_col_arg, cmap=cmap, legend=legend if plot_col_arg else False, 
             alpha=0.75, edgecolor='black', **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel("经度 (Longitude)")
    ax.set_ylabel("纬度 (Latitude)")
    
    final_title = title
    if add_basemap:
        try:
            gdf_for_basemap = gdf.to_crs(basemap_crs) # 确保GDF与底图CRS一致
            minx, miny, maxx, maxy = gdf_for_basemap.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            cx.add_basemap(ax, crs=gdf_for_basemap.crs.to_string(), source=basemap_source, attribution_size=5)
            final_title = f"{title} (含底图)"
            ax.set_title(final_title) # 更新标题
        except Exception as e:
            print(f"添加底图失败: {e}。请确保安装了contextily且有网络连接。将显示无底图的图像。")
    
    plt.tight_layout()
    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png if save_filename_png.lower().endswith('.png') else f"{save_filename_png}.png")
        plt.savefig(full_save_path)
        print(f"GeoPandas图像已保存至: {full_save_path}")
    else:
        plt.show()
    plt.close(fig) # 关闭图像，防止在循环中使用时重复显示

# ==============================================================================
# Folium 地图创建函数 (Folium Map Creation Functions)
# ==============================================================================

def create_folium_map(center_location=None, zoom_start=6, tiles="OpenStreetMap", **map_kwargs):
    """
    创建一个基础的Folium地图对象。

    参数:
    center_location (list or tuple, optional): 地图中心点 [纬度, 经度]。如果None，则尝试自动确定或使用默认值。
    zoom_start (int, optional): 初始缩放级别。
    tiles (str, optional): 地图瓦片类型 (例如 "OpenStreetMap", "Stamen Terrain", "CartoDB positron")。
    **map_kwargs: 传递给 folium.Map() 的其他关键字参数。

    返回:
    folium.Map: 创建的Folium地图对象。
    """
    if center_location is None:
        center_location = [39.9, 116.4] # 默认为北京
    return folium.Map(location=center_location, zoom_start=zoom_start, tiles=tiles, **map_kwargs)

def add_gdf_to_folium(folium_map, gdf, layer_name=None, 
                      popup_cols=None, tooltip_cols=None, style_function=None, highlight_function=None,
                      marker_type='marker', # 'marker', 'circle_marker', 'geojson' (for polygons/lines)
                      heatmap_col=None, heatmap_radius=15, # For heatmap layer on points
                      **kwargs):
    """
    将GeoDataFrame的几何对象添加到Folium地图。

    参数:
    folium_map (folium.Map): 要添加图层的Folium地图对象。
    gdf (geopandas.GeoDataFrame): 包含地理数据的GeoDataFrame。
    layer_name (str, optional): 图层的名称 (用于LayerControl)。默认为GeoDataFrame的名称或类型。
    popup_cols (list of str, optional): 点击时在弹出窗口中显示的列名。
    tooltip_cols (list of str, optional): 悬停时在工具提示中显示的列名。
    style_function (function, optional): GeoJSON图层的样式函数。
    highlight_function (function, optional): GeoJSON图层的高亮函数。
    marker_type (str, optional): 对于点数据，指定标记类型:
                                 'marker' (标准folium.Marker), 
                                 'circle_marker' (folium.CircleMarker),
                                 'geojson' (将点也作为GeoJson处理，通常用于多边形/线，但点也可以)。
    heatmap_col (str, optional): 如果是点数据且marker_type不是'geojson'，指定此列作为热力图的权重。
    heatmap_radius (int, optional): 热力图点的半径。
    **kwargs: 根据marker_type传递给相应的folium函数 (folium.Marker, folium.CircleMarker, folium.GeoJson, HeatMap) 的参数。
    """
    if not isinstance(gdf, geopandas.GeoDataFrame) or gdf.empty:
        print("输入不是有效的GeoDataFrame或为空，无法添加到Folium地图。")
        return folium_map

    # 确保gdf是WGS84 (EPSG:4326)，Folium通常使用这个
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    actual_layer_name = layer_name if layer_name else "GeoData Layer"
    
    # 处理几何类型
    geom_type = gdf.geometry.geom_type.iloc[0] if not gdf.empty else None

    if geom_type == "Point":
        if marker_type == 'marker' or marker_type == 'circle_marker':
            for idx, row in gdf.iterrows():
                loc = [row.geometry.y, row.geometry.x]
                
                popup_html = None
                if popup_cols:
                    popup_html = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in popup_cols if col in row])
                
                tooltip_text = None
                if tooltip_cols:
                    tooltip_text = ", ".join([f"{col}: {row[col]}" for col in tooltip_cols if col in row])
                elif popup_cols: # 默认tooltip为popup的第一个字段
                    tooltip_text = f"{popup_cols[0]}: {row[popup_cols[0]]}" if popup_cols[0] in row else actual_layer_name

                if marker_type == 'marker':
                    folium.Marker(location=loc, popup=popup_html, tooltip=tooltip_text, **kwargs).add_to(folium_map)
                elif marker_type == 'circle_marker':
                    # CircleMarker的参数: radius, color, fill, fill_color, fill_opacity
                    radius = kwargs.pop('radius', 5) 
                    color = kwargs.pop('color', 'blue')
                    fill = kwargs.pop('fill', True)
                    folium.CircleMarker(location=loc, radius=radius, color=color, fill=fill, 
                                        popup=popup_html, tooltip=tooltip_text, **kwargs).add_to(folium_map)
            
            if heatmap_col and heatmap_col in gdf.columns:
                heat_data = [[point.geometry.y, point.geometry.x, point[heatmap_col]] 
                             for _, point in gdf.iterrows() if pd.notna(point[heatmap_col])]
                if heat_data:
                    HeatMap(heat_data, name=f"{actual_layer_name} - 热力图", radius=heatmap_radius, **kwargs).add_to(folium_map)
        
        elif marker_type == 'geojson': # 将点也作为GeoJson处理
             # GeoJson的参数: style_function, highlight_function, name, tooltip, popup
            folium.GeoJson(
                gdf.to_json(),
                name=actual_layer_name,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(fields=tooltip_cols if tooltip_cols else []),
                popup=folium.features.GeoJsonPopup(fields=popup_cols if popup_cols else []),
                **kwargs
            ).add_to(folium_map)

    elif geom_type in ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]:
        folium.GeoJson(
            gdf.to_json(), # GeoDataFrame可以直接转为GeoJSON字符串
            name=actual_layer_name,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(fields=tooltip_cols if tooltip_cols else gdf.columns.drop('geometry').tolist()[:3], aliases=tooltip_cols if tooltip_cols else None), # 默认显示前几列
            popup=folium.features.GeoJsonPopup(fields=popup_cols if popup_cols else gdf.columns.drop('geometry').tolist()[:3], aliases=popup_cols if popup_cols else None),
            **kwargs
        ).add_to(folium_map)
    else:
        print(f"不支持的几何类型: {geom_type}。无法添加到Folium地图。")

    return folium_map


def save_folium_map(folium_map, save_filename_html, add_layer_control=True):
    """
    保存Folium地图到HTML文件，并可选添加图层控制器。

    参数:
    folium_map (folium.Map): 要保存的Folium地图对象。
    save_filename_html (str): HTML文件的保存名 (例如 'my_map.html')。
    add_layer_control (bool, optional): 是否在保存前添加图层控制器。
    """
    if not isinstance(folium_map, folium.Map):
        print("输入不是有效的Folium地图对象。")
        return

    if add_layer_control:
        folium.LayerControl().add_to(folium_map)
        
    full_save_path = os.path.join(OUTPUT_DIR, save_filename_html if save_filename_html.lower().endswith('.html') else f"{save_filename_html}.html")
    try:
        folium_map.save(full_save_path)
        print(f"Folium地图已保存至: {full_save_path}")
    except Exception as e:
        print(f"保存Folium地图失败: {e}")

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def _create_sample_geodataframes_for_demo():
    """内部辅助函数，为演示创建样本GeoDataFrames。"""
    points_data = {
        '城市': ['北京', '上海', '广州', '成都'],
        '人口 (万)': [2154, 2487, 1867, 1600],
        'geometry': [Point(116.4074, 39.9042), Point(121.4737, 31.2304), Point(113.2644, 23.1291), Point(104.0668, 30.5728)]
    }
    gdf_points = create_gdf_from_dict(points_data, crs="EPSG:4326")

    poly_data = {
        '区域名称': ['长江三角洲', '珠江三角洲'],
        '经济规模 (万亿)': [21.6, 11.5], # 假设数据
        'geometry': [
            Polygon([(118, 32), (122, 32), (122, 29), (118, 29)]), # 简化示意范围
            Polygon([(112, 23.5), (115, 23.5), (115, 21.5), (112, 21.5)])
        ]
    }
    gdf_polygons = create_gdf_from_dict(poly_data, crs="EPSG:4326")
    return gdf_points, gdf_polygons

def run_geospatial_demos():
    """运行所有地理空间可视化演示函数。"""
    print(f"--- 地理空间数据分析与可视化接口化演示 (图表和地图将保存到 '{OUTPUT_DIR}' 目录) ---")

    gdf_points_demo, gdf_polygons_demo = _create_sample_geodataframes_for_demo()

    # 1. GeoPandas基本操作演示
    print("\n=== GeoPandas 基本操作演示 (点数据) ===")
    get_gdf_info(gdf_points_demo)
    get_gdf_head(gdf_points_demo)
    get_gdf_crs(gdf_points_demo)
    print("点数据质心:", get_gdf_centroid(gdf_points_demo)) # 打印质心系列

    print("\n=== GeoPandas 基本操作演示 (面数据) ===")
    get_gdf_info(gdf_polygons_demo)
    gdf_polygons_reprojected = reproject_gdf(gdf_polygons_demo, "EPSG:3857") # 重投影到Web Mercator
    get_gdf_crs(gdf_polygons_reprojected)
    print("面数据面积 (投影后):", get_gdf_area(gdf_polygons_reprojected))
    print("面数据质心 (原CRS):", get_gdf_centroid(gdf_polygons_demo))


    # 2. GeoPandas绘图演示
    print("\n=== GeoPandas 绘图演示 ===")
    plot_geopandas(gdf_points_demo, column_to_plot='人口 (万)', title='中国主要城市人口分布 (GeoPandas)',
                   legend=True, add_basemap=True, save_filename_png="demo_geopandas_points_basemap.png",
                   markersize=gdf_points_demo['人口 (万)'] / 50) # 根据人口调整点大小

    plot_geopandas(gdf_polygons_demo, column_to_plot='经济规模 (万亿)', title='主要经济区域示意图 (GeoPandas)',
                   cmap='OrRd', save_filename_png="demo_geopandas_polygons.png", legend=True)

    # 3. Folium地图演示
    print("\n=== Folium 交互式地图演示 ===")
    # 创建基础地图
    folium_demo_map = create_folium_map(center_location=[35, 110], zoom_start=4) # 中国大致中心

    # 添加点数据图层
    folium_demo_map = add_gdf_to_folium(folium_demo_map, gdf_points_demo, layer_name='主要城市',
                                        popup_cols=['城市', '人口 (万)'], tooltip_cols=['城市'],
                                        marker_type='circle_marker', radius=8, color='crimson', fill_color='red',
                                        heatmap_col='人口 (万)', heatmap_radius=25)
    
    # 添加面数据图层
    def poly_style_func(feature):
        scale = feature['properties']['经济规模 (万亿)']
        return {
            'fillColor': 'green' if scale > 15 else 'yellow',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.5
        }
    folium_demo_map = add_gdf_to_folium(folium_demo_map, gdf_polygons_demo, layer_name='经济区域',
                                        popup_cols=['区域名称', '经济规模 (万亿)'], tooltip_cols=['区域名称'],
                                        style_function=poly_style_func)
    
    save_folium_map(folium_demo_map, "demo_folium_combined_map.html", add_layer_control=True)

    # 演示从文件读取 (需要一个示例文件，这里只是概念性代码)
    # print("\n=== 从文件读取GeoDataFrame演示 (概念性) ===")
    # example_geojson_path = "path_to_your_data.geojson" # 替换为真实路径
    # if os.path.exists(example_geojson_path):
    #     gdf_from_file = read_gdf_from_file(example_geojson_path)
    #     if not gdf_from_file.empty:
    #         get_gdf_info(gdf_from_file)
    #         plot_geopandas(gdf_from_file, title="从文件读取的数据", save_filename_png="demo_gdf_from_file.png")
    # else:
    #     print(f"示例文件 {example_geojson_path} 未找到，跳过文件读取演示。")


    print(f"--- 地理空间演示完成。输出保存在 '{OUTPUT_DIR}' 目录。 ---")

if __name__ == '__main__':
    run_geospatial_demos() 