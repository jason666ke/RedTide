#!/bin/bash

# 设置基础路径
BASE_DIR="$(pwd)"
DATA_PROCESS_DIR="${BASE_DIR}/data_process"
TASK1_DIR="${BASE_DIR}/task_1/Prediction_ModernTCN"
TASK2_DIR="${BASE_DIR}/task_2/detection"

# 创建日志目录
mkdir -p "${BASE_DIR}/logs"
LOG_FILE="${BASE_DIR}/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# 检查上一个命令是否成功
check_status() {
    if [ $? -ne 0 ]; then
        log "错误：$1 失败"
        exit 1
    fi
}

# 1. 执行数据预处理
log "开始数据预处理..."
cd "$DATA_PROCESS_DIR"
python main.py
check_status "数据预处理"
log "数据预处理完成"

# 2. 执行任务一（预测）
log "开始执行任务一..."
cd "$TASK1_DIR"

# 获取所有需要处理的站点
STATIONS=(
    "大亚湾坝光"
    "大亚湾长湾"
    "大亚湾东山"
    "大亚湾东冲"
    "大鹏湾沙头角"
    "大鹏湾大梅沙"
    "大鹏湾下沙"
    "大鹏湾南澳"
    "大鹏湾湾口"
    "珠江口沙井"
    "深圳湾蛇口"
    "珠江口矾石"
    "珠江口内伶仃南"
)

# 遍历处理每个站点
for station in "${STATIONS[@]}"; do
    log "处理站点: $station"
    
    # 修改predict.sh中的站点名称
    sed -i "s/^MODEL=.*/MODEL=\"$station\"/" predict.sh
    sed -i "s|^DATA_FILE=.*|DATA_FILE=\"../data_9features/${station}.csv\"|" predict.sh
    
    # 执行预测脚本
    ./predict.sh
    check_status "任务一：站点 $station 的预测"
    
    log "完成站点 $station 的预测"
done

log "任务一执行完成"

# 3. 执行任务二（异常检测）
log "开始执行任务二..."
cd "$TASK2_DIR"

# 执行异常检测脚本
./map_station_run.sh
check_status "任务二：异常检测"

log "任务二执行完成"

log "整个流程执行完成"

# 显示汇总信息
log "执行汇总："
log "- 数据预处理输出目录: ${BASE_DIR}/task_1/data_9features/"
log "- 任务一输出目录: ${BASE_DIR}/task_2/data_final/"
log "- 任务二输出目录: ${BASE_DIR}/task_2/results/"