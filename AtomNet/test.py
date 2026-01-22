import requests
proxies = {
    "http": "http://127.0.0.1:10808",
    "https": "http://127.0.0.1:10808",
}
try:
    r = requests.get("https://api.wandb.ai", proxies=proxies, timeout=10)
    print("连接成功 ✅")
except Exception as e:
    print("连接失败 ❌", e)
