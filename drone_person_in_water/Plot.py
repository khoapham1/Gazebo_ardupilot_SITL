import pandas as pd
import matplotlib.pyplot as plt

data_str = """23.00
14.00
16.00
8.00
5.00
47.00
62.00
121.00
72.00
38.00
76.00
3.00
28.00
20.00
54.00
70.00
18.00
75.00
78.00
74.00
55.00
20.00
80.00
66.00
53.00
10.00
34.00
46.00
13.00
4.00"""

minutes = [float(x.strip()) for x in data_str.splitlines() if x.strip() != ""]

df = pd.DataFrame({
    "case": list(range(1, len(minutes)+1)),
    "minutes": minutes
})

# Đảo ngược dữ liệu (cho vùng trước tiêm nằm dưới 0)
df["minutes_inverted"] = -df["minutes"]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["case"], df["minutes_inverted"], marker='o', color="red")

ax.set_ylim(-150, 150)
ax.set_yticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])

## Ten label
ax.set_ylabel("Phút (minutes, inverted for pre-injection)")
ax.set_xlabel("Case (số trường hợp)")
ax.set_title("Số phút trước/sau tiêm (0 ở giữa, mở rộng ±150)")

# Đường ngang
ax.axhline(0, linewidth=1.2, linestyle='--')

# Lưới
ax.grid(True, linestyle=':', linewidth=0.5)

# Lưu file hình
plt.savefig("minutes_plot.png", bbox_inches='tight')

# Hiển thị trên màn hình
plt.show()
