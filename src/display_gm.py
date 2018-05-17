# -*- coding:utf-8 -*-
import eviltransform as ets

const_prefix = (
    "<!DOCTYPE html>\n"
    "<html>\n"
    "<head>\n"
    "<title>Simple Map</title>\n"
    "<meta name=\"viewport\" content=\"initial-scale=1.0\">\n"
    "<meta charset=\"utf-8\">\n"
    "<style>\n"
    "#map {\n"
    "height: 100%%;\n"
    "}\n"
    "html, body {\n"
    "height: 100%%;\n"
    "margin: 0;\n"
    "padding: 0;\n"
    "}\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<div id=\"map\"></div>\n"
    "<script>\n"
    "function initMap() {\n"
    "var map = new google.maps.Map(document.getElementById('map'), {\n"
    "zoom: 16,\n"
    "center: {lat: %s, lng: %s},\n"
    "mapTypeId: 'terrain'\n"
    "});\n")
const_suffix = (
    "}\n"
    "</script>\n"
    "<script async defer src=\"https://maps.googleapis.com/maps/api/js?key=AIzaSyAK4J1HJ8jeSyuYXWpcdziGJXOqw7xP6Y8&callback=initMap\">\n"
    "</script>\n"
    "</body>\n"
    "</html>\n"
)
const_marker = (
    "var marker_%d = new google.maps.Marker({"
    "position: %s,"
    "map: map"
    "});\n"
)

const_marker_title = (
    "var marker_%d = new google.maps.Marker({"
    "position: %s,"
    "map: map,"
    "title: '%s'"
    "});\n"
)

const_marker_info = "addInfoW(marker_%d, '%s');\n\n"

const_path = (
    "var path_%d = new google.maps.Polyline({\n"
    "path: %s,\n"
    "geodesic: true,\n"
    "strokeColor: '%s',\n"
    "strokeOpacity: 0.6,\n"
    "strokeWeight: 2\n"
    "});\n"
)
const_circle = (
    "var circle_%d = new google.maps.Circle({\n"
    "strokeColor: '%s',\n"
    "strokeWeight: 2,\n"
    "fillColor: '%s',\n"
    "fillOpacity: 0.35,\n"
    "map: map,\n"
    "center: %s,\n"
    "radius: %s\n"
    "});\n"
)

const_coor = (
    "var coor_%d = %s;\n"
)
colors = [
    '#000000',  # 黑色
    '#FF0000',  # 红色
    '#00FF00',  # 绿色
    '#FFFF00',  # 黄色
    '#0000FF',  # 蓝色
]


def pd_to_html(df, out_fname):
    c_lng, c_lat = df.iloc[0]['Longitude'], df.iloc[0]['Latitude']
    html = const_prefix % (c_lat, c_lng)
    for i in range(len(df)):
        lng, lat = df.iloc[i]['Longitude'], df.iloc[i]['Latitude']
        # print 'O:(%s, %s)' % (lat, lng)
        lat, lng = ets.wgs2gcj(lat, lng)
        # print 'N:(%s, %s)' % (lat, lng)
        marker = '{lat:%s, lng:%s}' % (lat, lng)
        html += const_marker % (i, marker)
    html += const_suffix
    with open(out_fname, 'w') as o_file:
        o_file.write(html)

def edges_to_html(edges, out_fname):
    c_lat, c_lng = edges[edges.keys()[0]][0]
    html = const_prefix % (c_lat, c_lng)
    rid = 0
    for e_id, points in edges.iteritems():
        p_str = []
        coors = '['
        for idx, point in enumerate(points):
            lat, lng = float(point[0]), float(point[1])
            lat, lng = ets.wgs2gcj(lat, lng)
            marker = '{lat:%s, lng:%s}' % (lat, lng)
            p_str.append(marker)
        coors += ', '.join(p_str) + ']\n'
        html += const_coor % (rid, coors)
        html += const_path % (rid, 'coor_' + str(rid), colors[0])
        html += 'path_%d' % rid + '.setMap(map);\n\n'
        rid += 1
    html += const_suffix
    with open(out_fname, 'w') as o_file:
        o_file.write(html)

def mr_cell_to_html(obsv, cells, out_fname):
    c_lat, c_lng = obsv[0][0], obsv[0][1]
    html = const_prefix % (c_lat, c_lng)
    html += (
        "function addInfoW(marker, message){\n"
        "var infoW = new google.maps.InfoWindow({content:message});\n"
        "google.maps.event.addListener(marker, 'click', function () {infoW.open(map, marker);});\n"
        "}\n"
    )
    for rid, point in enumerate(obsv):
        lat, lng = ets.wgs2gcj(point[0], point[1])
        marker = '{lat:%s, lng:%s}' % (lat, lng)
        html += const_marker % (rid, marker)
        html += const_marker_info % (rid, '%d.0 %d.0' % (point[2], point[3]))
    for ids, info in cells.iteritems():
        lat, lng = ets.wgs2gcj(info[1], info[2])
        marker = '{lat:%s, lng:%s}' % (lat, lng)
        idx = int(info[0])
        if idx == -1:
            continue
        html += const_circle % (idx, colors[idx % (len(colors))], colors[idx % (len(colors))], marker, 50)
        html += const_marker_title % (idx + rid, marker, 'Cell Tower (%s,%s)' %(ids[0], ids[1]))
        html += const_marker_info % (idx + rid, '[%d]' % idx)
        # html += 'marker_' + str(idx + rid) + '.click();\n\n'
        # print idx, '\t', ids
    html += const_suffix
    with open(out_fname, 'w') as o_file:
        o_file.write(html)
