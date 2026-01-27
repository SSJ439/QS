# 测试 API 连接的代码
import requests

def test_connection():
    # 使用你之前提取的 get_stock_list 接口
    url = "http://192.168.1.10:10213/api/get_stock_list"
    params = {"date": "20251225"} # 示例日期
    try:
        response = requests.get(url, params=params)
        print(f"接口状态码: {response.status_code}")
        print(f"数据预览: {response.json()[:5]}") # 只看前5条
    except Exception as e:
        print(f"连接失败: {e}")

if __name__ == "__main__"

    # 桑科
