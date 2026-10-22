import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import math

class TurtlePolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

class GoalInferenceNode(Node):
    def __init__(self):
        super().__init__('goal_inference_node')
        
        # ハイパーパラメータ・設定値
        self.map_size = 11.0
        self.max_linear = 2.0
        self.max_angular = 2.0
        self.goal_x = 5.5
        self.goal_y = 5.5
        
        # モデルのロード
        self.device = torch.device('cpu')
        self.model = TurtlePolicyNetwork()
        
        try:
            checkpoint = torch.load('turtle_goal_policy.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info("Successfully loaded 'turtle_goal_policy.pth'.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model weight: {e}")
            
        self.model.eval()
        
        # ROS 2 サブスクライバーとパブリッシャー
        self.subscription = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.pose_callback,
            10
        )
        self.publisher = self.create_publisher(
            Twist,
            '/turtle1/cmd_vel',
            10
        )

    def pose_callback(self, msg):
        x = msg.x
        y = msg.y
        theta = msg.theta
        
        # 固定ゴールとの相対距離
        goal_dx = self.goal_x - x
        goal_dy = self.goal_y - y
        
        # 正規化
        x_norm = x / self.map_size
        y_norm = y / self.map_size
        goal_dx_norm = goal_dx / self.map_size
        goal_dy_norm = goal_dy / self.map_size
        
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        # 入力テンソル作成 (6次元)
        state = torch.tensor([
            x_norm,
            y_norm,
            sin_theta,
            cos_theta,
            goal_dx_norm,
            goal_dy_norm
        ], dtype=torch.float32).unsqueeze(0)
        
        # 推論実行
        with torch.no_grad():
            action = self.model(state).squeeze(0)
            
        # 非正規化（実速度に変換）
        linear_vel = action[0].item() * self.max_linear
        angular_vel = action[1].item() * self.max_angular
        
        # 終了判定（ゴール付近に到達したら停止）
        dist = math.sqrt(goal_dx**2 + goal_dy**2)
        twist = Twist()
        
        if dist < 0.15:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("Goal reached! Stopping turtle.", throttle_duration_sec=2.0)
        else:
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)
            
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = GoalInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
