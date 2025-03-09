#!/bin/bash

# 声明关联数组（字典）用于存储站点映射关系
declare -A STATION_MAPPING

# 定义站点名称到代码的映射
STATION_MAPPING=(
    ["大亚湾坝光"]="dywbg"
    ["大亚湾长湾"]="dywcw"
    ["大亚湾东山"]="dywds"
    ["大亚湾东冲"]="dywdy"
    ["大鹏湾沙头角"]="dpwstj"
    ["大鹏湾大梅沙"]="dpw_dameisha"
    ["大鹏湾下沙"]="dpwxs"
    ["大鹏湾南澳"]="dpw_nanao"
    ["大鹏湾湾口"]="dpwwk"
    ["珠江口沙井"]="zjksj"
    ["深圳湾蛇口"]="szwsk"
    ["珠江口矾石"]="zjkfs"
    ["珠江口内伶仃南"]="zjkln"
)

# 导出映射数组，使其他脚本可以使用
export STATION_MAPPING

# 定义一个函数用于获取站点代码
get_station_code() {
    local station_name="$1"
    echo "${STATION_MAPPING[$station_name]}"
}

# 定义一个函数用于列出所有站点映射
list_all_stations() {
    for station in "${!STATION_MAPPING[@]}"; do
        echo "$station -> ${STATION_MAPPING[$station]}"
    done
}

# 如果直接运行此脚本，显示所有站点映射关系
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "站点映射关系："
    echo "----------------"
    list_all_stations
fi