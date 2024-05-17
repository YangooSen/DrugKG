#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <任务名称> <要运行的命令>"
    exit 1
fi

# 从命令行参数中获取任务名称和要运行的命令
task_name="$1"
command_to_run="${*:2}"

# 定义日志文件和错误文件的路径
log_file="$task_name.log"
error_file="$task_name.err"

# 使用nohup运行任务，并将输出和错误重定向到log文件和error文件
nohup $command_to_run > $log_file 2> $error_file &

# 打印任务已启动的消息
echo "任务 '$task_name' 已启动，日志文件保存在 $log_file，错误文件保存在 $error_file"
