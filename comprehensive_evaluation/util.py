import pandas as pd
import numpy as np
import os
import sys


def transform_market_order_strategy(
    data, positions, action_dim=5, max_holding_number=0.01
):
    previous_position = 0
    timelist = data["timestamp"].tolist()[:-1]
    assert len(timelist) == len(positions)
    strategy = []
    previous_position = 0
    for timestamp, position in zip(timelist, positions):
        if previous_position != position:
            size = (
                abs(position - previous_position)
                / (action_dim - 1)
                * max_holding_number
            )
            action = "buy" if position > previous_position else "sell"
            order = []
            if action == "buy":
                ask_prices = data[data["timestamp"] == timestamp][
                    [
                        "ask1_price",
                        "ask2_price",
                        "ask3_price",
                        "ask4_price",
                        "ask5_price",
                    ]
                ].values.tolist()[0]
                ask_sizes = data[data["timestamp"] == timestamp][
                    ["ask1_size", "ask2_size", "ask3_size", "ask4_size", "ask5_size"]
                ].values.tolist()[0]
                for price, ask_size in zip(ask_prices, ask_sizes):
                    if size > 0:
                        if size > ask_size:
                            order.append({"price": price, "amount": ask_size})
                            size -= ask_size
                        else:
                            order.append({"price": price, "amount": size})
                            size = 0
            if action == "sell":
                bid_prices = data[data["timestamp"] == timestamp][
                    [
                        "bid1_price",
                        "bid2_price",
                        "bid3_price",
                        "bid4_price",
                        "bid5_price",
                    ]
                ].values.tolist()[0]
                bid_sizes = data[data["timestamp"] == timestamp][
                    ["bid1_size", "bid2_size", "bid3_size", "bid4_size", "bid5_size"]
                ].values.tolist()[0]
                for price, bid_size in zip(bid_prices, bid_sizes):
                    if size > 0:
                        if size > bid_size:
                            order.append({"price": price, "amount": bid_size})
                            size -= bid_size
                        elif size != 0:
                            order.append({"price": price, "amount": size})
                            size = 0
            previous_position = position
            strategy.append(
                {
                    "timestamp": timestamp,
                    "action": action,
                    "order": order,
                    "position": position / (action_dim - 1) * max_holding_number,
                }
            )
    return strategy


if __name__ == "__main__":
    positions = np.load("data/micro_action.npy")
    data = pd.read_feather("data/test.feather")
    print(len(positions))
    print(len(data))
    strategy=transform_market_order_strategy(data, positions,max_holding_number=4000)
    print(strategy)
