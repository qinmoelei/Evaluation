import pandas as pd
import numpy as np
#主要看valid 和 test的区别以及每个agent的分布的柱状图
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
action_BTCTUSD=np.load("/home/mlqin/qml/HFT_06_17/result_risk/BTCTUSD/high_level/seed_12345/epoch_54/test/action.npy")
action_BTCUSDT=np.load("/home/mlqin/qml/HFT_06_17/result_risk/BTCUSDT/high_level/seed_12345/epoch_42/test/action.npy")
action_ETHUSDT=np.load("/home/mlqin/qml/HFT_06_17/result_risk/ETHUSDT/high_level/seed_12345/epoch_58/test/action.npy")
action_GALAUSDT=np.load("/home/mlqin/qml/HFT_06_17/result_risk/GALAUSDT/high_level/seed_12345/epoch_27/test/action.npy")
market_trend_freq_list_list=[]
for market,action in zip([0,1,2,3],[action_BTCTUSD,action_BTCUSDT,action_ETHUSDT,action_GALAUSDT]):
    market_trend_freq_list=[0]*5
    for market_trend in action:
        market_trend_freq_list[market_trend]=market_trend_freq_list[market_trend]+1/len(action)
    market_trend_freq_list=[round(i,4) for i in market_trend_freq_list]
    market_trend_freq_list_list.append(market_trend_freq_list)
market_trend_freq_list_list=np.array(market_trend_freq_list_list)*100

x_label = ["Bull", "Rally", "Sideways", "Pullback", "Bear"]
x_label.reverse()

x = np.array(range(len(x_label)))*2
color_list = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#BEB8DC"]
y_label = ["BTCT", "BTCU", "ETH", "GALA"]

plt.figure(figsize=(10, 7))
for i in range(4):
    plt.bar(
        [m + i * 0.4 for m in x],
        market_trend_freq_list_list[i],
        label=y_label[i],
        width=0.4,
        color=color_list[i],
    )

plt.xticks([i + 0.8 for i in x], x_label, fontsize="x-large")

plt.ylabel("Selected Percentage (%)", fontsize="x-large")
plt.xlabel("Market Trends", fontsize="x-large")
patches = [
    mpatches.Patch(color=color_list[i], label="{:s}".format(y_label[i]))
    for i in range(len(color_list))
]
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.75])
ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize='large')
for i in range(len(market_trend_freq_list_list)):
    data=market_trend_freq_list_list[i]
    for x,y in enumerate(data):
        plt.text(x*2+i*0.4, y+0.5, round(y,1),ha='center',fontsize=9)

# plt.tight_layout()
plt.savefig("analysis/plot/market_trend_freq.pdf",bbox_inches='tight')
plt.show()