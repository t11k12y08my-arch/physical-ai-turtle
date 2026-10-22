# physical-ai-turtle
Physical AI 応用１講座　最終課題用のリポジトリ
Overview
Turtlesim上でBehavior Cloningを用いて目標地点へ自律移動するAIを構築した。

Training
ROS2 Bagからデモデータを収集し、PyTorchで学習を行った。

Input
[x, y, sin(theta), cos(theta), goal_dx, goal_dy]

Output
[linear.x, angular.z]

Goal
(5.5, 5.5)

Environment
Ubuntu 22.04
ROS2 Humble
PyTorch
rosbags
Run
Training
python3 train.py

Inference
python3 inference.py
