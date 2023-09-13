import pandas as pd
import numpy as np
import warnings
import sys
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(".")
from tool import market_dynamics_modeling_analysis
from tool import label_util as util
from comprehensive_evaluation.analyzer import *
from comprehensive_evaluation.util import *
from comprehensive_evaluation.slice_model import *
import bisect


def find_previous_element(sorted_list, value):
    index = bisect.bisect_left(sorted_list, value)
    if index == 0:
        return None  # 给定值小于列表中的所有元素
    else:
        return sorted_list[index - 1]


class Analyzer:
    def __init__(self, market_information, strategy,commission_rate=0.00015):
        # market information should be a data frame that consist of 21 columns(timedstamp &  5-level ask bid's price and size)
        # stategy: a list of dictionary consisting of 3 keys: timestamp, action (buy or sell) and dictionary indicating the level
        # price and amount of the conducted orders
        self.market_information = market_information
        self.commission_rate=commission_rate
        self.strategy = strategy
        # order in strategy[{'timestamp':,'action':,'order':[{'price':,'amount':},{'price':,'amount':},...],'position':}]
        # the strategy should be a list of market order containing the executed price, amount the agent's position after conducting the trade
        # check the pricing the problem
        price_timing = [price["timestamp"] for price in self.strategy]
        assert max(price_timing) <= max(self.market_information.timestamp)
        # check the price is legal
        for timestamp in self.market_information.timestamp.unique():
            price_single_timestamp = self.market_information[
                self.market_information["timestamp"] == timestamp
            ]

            assert (
                price_single_timestamp["ask1_price"].values[0]
                >= price_single_timestamp["bid1_price"].values[0]
            )
        # check the strategy opening position is whether is over rated and place correctly
        # 对于买单我们要检查他的买入价格的下限应该符合ask1 price
        for stack_order in strategy:
            timestamp = stack_order["timestamp"]
            current_market_information = self.market_information[
                self.market_information["timestamp"] == timestamp
            ]
            assert stack_order["action"] in ["buy", "sell"]
            if stack_order["action"] == "buy":
                list_order = stack_order["order"]
                level_number = len(list_order)
                for i in range(level_number):
                    assert (
                        list_order[i]["price"]
                        == current_market_information[
                            "ask{}_price".format(i + 1)
                        ].values[0]
                    )
                    assert (
                        list_order[i]["amount"]
                        <= current_market_information[
                            "ask{}_size".format(i + 1)
                        ].values[0]
                    )
            elif stack_order["action"] == "sell":
                list_order = stack_order["order"]
                level_number = len(list_order)
                for i in range(level_number):
                    assert (
                        list_order[i]["price"]
                        == current_market_information[
                            "bid{}_price".format(i + 1)
                        ].values[
                            0
                        ]  # ？？？？？？？？？bid0_price  format(i)
                    )
                    assert (
                        list_order[i]["amount"]
                        <= current_market_information[
                            "bid{}_size".format(i + 1)
                        ].values[0]
                    )
        # check the trace of the position in the trading process is legal or not. it always should be 0 at the start and end of the trading process
        if self.strategy[-1]["position"] != 0:
            last_position = self.strategy[-1]["position"]
            warnings.warn(
                "the final position of the strategy is not zero, we force the agent to close its position in the last timestamp"
            )
            last_market_information = self.market_information[
                self.market_information["timestamp"]
                == max(self.market_information["timestamp"].unique())
            ]
            size_sum = 0
            if (
                last_position
                > last_market_information["bid1_size"].values[0]
                + last_market_information["bid2_size"].values[0]
                + last_market_information["bid3_size"].values[0]
                + last_market_information["bid4_size"].values[0]
                + last_market_information["bid5_size"].values[0]
            ):
                warnings.warn(
                    "we barely trade at this timstamp instantly because there is no enough liquidity in the market,\
                we force the agent to close its position in the last timestamp by expanding the last level's size"
                )
                last_market_information["bid5_size"] = last_position - (
                    last_market_information["bid1_size"].values[0]
                    + last_market_information["bid2_size"].values[0]
                    + last_market_information["bid3_size"].values[0]
                    + last_market_information["bid4_size"].values[0]
                )
            for i in range(5):
                size_sum += last_market_information["bid{}_size".format(i + 1)].values[
                    0
                ]
                if last_position <= size_sum:
                    break
            level_order_size_list = []
            order_remaining = last_position
            for j in range(i + 1):
                level_order_size_list.append(
                    {
                        "price": last_market_information[
                            "bid{}_price".format(j + 1)
                        ].values[0],
                        "amount": min(
                            order_remaining,
                            last_market_information["bid{}_size".format(j + 1)].values[
                                0
                            ],
                        ),
                    }
                )
                order_remaining = (
                    order_remaining
                    - last_market_information["bid{}_size".format(j + 1)].values[0]
                )
            self.strategy.append(
                {
                    "timestamp": last_market_information["timestamp"].values[0],
                    "action": "sell",
                    "order": level_order_size_list,
                    "position": 0,
                }
            )

    def analysis_behavior(self, selected_strategy):
        # 现确定总共的开闭仓的次数 selected strategy 起码开头和结尾的position应该为0
        opening_strategy_timestamp_list = []
        closing_strategy_timestamp_list = []

        for stack_order in selected_strategy:
            if stack_order["action"] == "buy":
                order_size = 0
                for order in stack_order["order"]:
                    order_size += order["amount"]
                if order_size == stack_order["position"]:
                    opening_strategy_timestamp_list.append(stack_order["timestamp"])
            elif stack_order["action"] == "sell":
                if stack_order["position"] == 0:
                    closing_strategy_timestamp_list.append(stack_order["timestamp"])
        assert len(opening_strategy_timestamp_list) == len(
            closing_strategy_timestamp_list
        )  # 确保开仓和平仓的次数相同
        trade_timestamp_list = list(
            zip(opening_strategy_timestamp_list, closing_strategy_timestamp_list)
        )

        # 1. 计算每次交易的收益率以及开仓到平仓的时间
        total_return_rate = 0
        total_duration = timedelta()
        total_mdd = 0
        count_pos_return_rate = 0

        for open_time, close_time in trade_timestamp_list:
            assert open_time < close_time
            single_trade_strategy = []
            for selected_stack_order in selected_strategy:
                if (
                    selected_stack_order["timestamp"] >= open_time
                    and selected_stack_order["timestamp"] <= close_time
                ):
                    single_trade_strategy.append(selected_stack_order)
            cash_flow = []
            # print("single_trade_strategy", single_trade_strategy)
            # 计算每次交易的现金流的变化
            for stack_order in single_trade_strategy:
                # print("stack_order", stack_order)
                total_value = 0
                for order in stack_order["order"]:
                    total_value += order["price"] * order["amount"]
                if stack_order["action"] == "buy":
                    total_value = -total_value
                    total_value=total_value*(1+self.commission_rate)
                if stack_order["action"] == "sell":
                    total_value=total_value*(1-self.commission_rate)
                cash_flow.append(total_value)
            cash_record = [sum(cash_flow[: i + 1]) for i in range(len(cash_flow))]
            final_cash = cash_record[-1]
            require_money = -min(cash_record)
            # 计算每次交易的收益率和持仓时间
            return_rate = final_cash / require_money
            total_return_rate += return_rate

            if return_rate > 0:
                count_pos_return_rate += 1
                total_duration += close_time - open_time

            total_duration += close_time - open_time
            # TODO 根据bid1 price进行结算，每次持仓过程中的maxdrawdown
            position_record = []
            timestamp_record = []
            trade_position_record = []
            cash_accmulative_record = []
            for stack_order in single_trade_strategy:
                timestamp_record.append(stack_order["timestamp"])
                position_record.append(stack_order["position"])
            corresponding_market_timestamp = [
                timestamp
                for timestamp in self.market_information["timestamp"].unique()
                if timestamp >= open_time and timestamp <= close_time
            ]
            assert len(timestamp_record) == len(position_record)
            assert len(timestamp_record) == len(cash_record)

            for i in range(len(timestamp_record) - 1):
                time_point = [
                    timestamp
                    for timestamp in self.market_information["timestamp"].unique()
                    if (
                        timestamp >= timestamp_record[i]
                        and timestamp < timestamp_record[i + 1]
                    )
                ]
                cash_accmulative_record.append(cash_record[i] + require_money)
                for j in range(len(time_point)):
                    trade_position_record.append(position_record[i])
                for k in range(len(time_point)):
                    cash_accmulative_record.append(cash_accmulative_record[-1])
            trade_position_record.append(0)
            cash_accmulative_record.append(cash_record[i + 1] + require_money)
            # print("cash_accmulative_record", cash_accmulative_record)

            # trade_position_record.append(0)
            corresponding_market_information = self.market_information[
                self.market_information["timestamp"].isin(
                    corresponding_market_timestamp
                )
            ]
            # print("trade_position_record_length", len(trade_position_record))
            # print("corresponding_market_information_length", len(corresponding_market_information))
            # if len(trade_position_record) != len(corresponding_market_information):
            #     print("trade_position_record", trade_position_record)
            #     print("corresponding_market_information", corresponding_market_information)

            assert len(trade_position_record) == len(corresponding_market_information)

            position_value_record = [
                position * single_value
                for position, single_value in zip(
                    trade_position_record,
                    corresponding_market_information["bid1_price"].values,
                )
            ]
            total_value_record = [
                cash + position_value
                for cash, position_value in zip(
                    cash_accmulative_record, position_value_record
                )
            ]
            mdd = 0
            peak = total_value_record[0]
            for value in total_value_record:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > mdd:
                    mdd = dd
            total_mdd += mdd

        mean_return_rate = total_return_rate / len(trade_timestamp_list)
        mean_duration = total_duration / len(trade_timestamp_list)
        mean_mdd = total_mdd / len(trade_timestamp_list)
        # TODO 你还需要计算这段策略中的胜率（trade中return rate大于0的数量/trade的总数量）
        win_rate = count_pos_return_rate / len(trade_timestamp_list)
        return mean_return_rate, mean_duration, mean_mdd, win_rate

    def calculate_metric(self, selected_timestamp: list):
        # selected trade is part of the strategy that we want to calculate the metric,
        # its position do not have to end or start with 0
        # selected_timestamp is a 2-element list indicating the start and end of the timestamp
        selected_timestamp.sort()
        assert len(selected_timestamp) == 2
        selected_market = self.market_information[
            (
                self.market_information["timestamp"]
                >=selected_timestamp[0]
            )
            & (self.market_information["timestamp"] <= selected_timestamp[1])
        ]
        selected_strategy = [
            item
            for item in self.strategy
            if selected_timestamp[0] <= item["timestamp"] <= selected_timestamp[1]
        ]
        first_trade = selected_strategy[0]
        first_trade_size = sum([level["amount"] for level in first_trade["order"]])
        if first_trade["action"] == "buy":
            first_trade_size = -first_trade_size

        initial_posotion = selected_strategy[0]["position"] + first_trade_size
        assert initial_posotion >= 0
        # 默认第一步自动补仓 用bid1买的 看啥情况
        cash_flow = [
            -initial_posotion
            * self.market_information[
                self.market_information["timestamp"] == selected_timestamp[0]
            ]["bid1_price"].values[0]
        ]
        for stack_order in selected_strategy:
            total_value = 0
            for order in stack_order["order"]:
                total_value += order["price"] * order["amount"]
                if stack_order["action"] == "buy":
                    total_value = -total_value
                    total_value=total_value*(1+self.commission_rate)
                if stack_order["action"] == "sell":
                    total_value=total_value*(1-self.commission_rate)
                cash_flow.append(total_value)
                cash_flow.append(total_value)

        cash_record = [sum(cash_flow[: i + 1]) for i in range(len(cash_flow))]
        # print("cash_record", cash_record)
        # cash record 总计来讲现金流的list
        require_money = -min(cash_record)
        # 最小的现金流 （最缺钱的 就是require money）
        position = initial_posotion
        position_market_record = []
        cash_market_record = []
        for timestamp in selected_market.timestamp.unique():
            matching_strategy = None
            for item in selected_strategy:
                if item["timestamp"] == timestamp:
                    matching_strategy = item
                    break
            if matching_strategy:
                current_position = matching_strategy["position"]
                position = current_position
            else:
                current_position = position
            position_market_record.append(current_position)
        inital_cash = require_money - cash_flow[0]
        cash = inital_cash
        for timestamp in selected_market.timestamp.unique():
            matching_strategy = next(
                (item for item in selected_strategy if item["timestamp"] == timestamp),
                None,
            )
            if matching_strategy:
                total_value = 0
                for order in matching_strategy["order"]:
                    total_value += order["price"] * order["amount"]
                if matching_strategy["action"] == "buy":
                    total_value = -total_value
                    total_value=total_value*(1+self.commission_rate)
                if matching_strategy["action"] == "sell":
                    total_value=total_value*(1-self.commission_rate)
                cash_flow.append(total_value)
                current_cash = cash + total_value
                cash = current_cash
                # print('total_value',total_value)
            else:
                current_cash = cash
            # print("current_cash", current_cash)
            cash_market_record.append(current_cash)
        assert len(position_market_record) == len(cash_market_record)
        # print(len(selected_market))
        # print(len(position_market_record))
        assert len(position_market_record) == len(selected_market)
        selected_market_price = selected_market["bid1_price"].values
        position_value_record = [
            position * single_value
            for position, single_value in zip(
                position_market_record, selected_market_price
            )
        ]
        # print("require_money", require_money)

        total_value_record = [
            cash + position_value
            for cash, position_value in zip(cash_market_record, position_value_record)
        ]
        # print("cash", cash_market_record)
        tr = total_value_record[-1] / total_value_record[0] - 1
        mdd = 0
        peak = total_value_record[0]
        for value in total_value_record:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
        cr = tr / mdd
        return tr, mdd, cr

    def analysis_along_time(self, num_seg):
        opening_strategy_timestamp_list = []
        closing_strategy_timestamp_list = []

        for stack_order in self.strategy:
            if stack_order["action"] == "buy":
                order_size = 0
                for order in stack_order["order"]:
                    order_size += order["amount"]
                if order_size == stack_order["position"]:
                    opening_strategy_timestamp_list.append(stack_order["timestamp"])
            elif stack_order["action"] == "sell":
                if stack_order["position"] == 0:
                    closing_strategy_timestamp_list.append(stack_order["timestamp"])
        assert len(opening_strategy_timestamp_list) == len(
            closing_strategy_timestamp_list
        )
        print(len(opening_strategy_timestamp_list))

        assert len(opening_strategy_timestamp_list) >= num_seg
        assert num_seg > 0
        num_trade_seg = int(len(opening_strategy_timestamp_list) / num_seg)
        market_timestamp_list = self.market_information.timestamp.unique().tolist()
        if num_seg == 1:
            selected_timestamp = [
                market_timestamp_list[0],
                market_timestamp_list[-1],
            ]
            tr, mdd, cr = self.calculate_metric(selected_timestamp)
            print(
                "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                    0, tr, mdd, cr
                )
            )
        else:
            for i in range(num_seg):
                if i == 0:
                    # 从这个一个segement的trade的结束下一步到这个segement的平仓算完
                    selected_timestamp = [
                        market_timestamp_list[0],
                        find_previous_element(
                            market_timestamp_list,
                            opening_strategy_timestamp_list[num_trade_seg * (i + 1)],
                        ),
                    ]
                    tr, mdd, cr = self.calculate_metric(selected_timestamp)

                    print(
                        "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                            i, tr, mdd, cr
                        )
                    )
                else:
                    if i != num_seg - 1:
                        selected_timestamp = [
                            opening_strategy_timestamp_list[num_trade_seg * i],
                            find_previous_element(
                                market_timestamp_list,
                                opening_strategy_timestamp_list[
                                    num_trade_seg * (i + 1)
                                ],
                            ),
                        ]

                        tr, mdd, cr = self.calculate_metric(selected_timestamp)
                        print(
                            "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                                i, tr, mdd, cr
                            )
                        )
                    else:
                        selected_timestamp = [
                            opening_strategy_timestamp_list[num_trade_seg * i],
                            market_timestamp_list[-1],
                        ]

                        tr, mdd, cr = self.calculate_metric(selected_timestamp)
                        print(
                            "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                                i, tr, mdd, cr
                            )
                        )

    def analysis_along_dynamics(self, num_dynamics, path, elected_timestamp: list):
        # TODO 给定段时间，先把这段时间内的市场状态划分成num_dynamics个市场状态，找到这段时间对应的策略（通过comprehensive_evaluation/slice_model.py，
        # TODO 你可能需要修改最后一部分 使得他知道他对应timestamp的dynamics的label是什么即可）
        # TODO 然后计算每个市场状态下策略开仓次数占开仓总次数的多少百分比，平仓占平仓总次数的百分比
        # TODO 然后计算每个市场状态下策略开仓量占开仓总量的多少百分比，平仓量占平仓总量的百分比
        # TODO 思路 找strategy中对应的timestamp（应该相互吻合，然后根据这个timestamp找到对应的dynamics的 然后做统计）
        pass
        # selected_market = data[
        #     (data["timestamp"] >= elected_timestamp[0])
        #     & (data["timestamp"] <= elected_timestamp[1])
        # ]
        # model = Linear_Market_Dynamics_Model(
        #     data=selected_market.reset_index(), dynamic_number=num_dynamics
        # )

        # model.run(path)

        # df_label = pd.read_feather(path + "/df_label.feather")
        # assert len(selected_market) == len(df_label)

        # opening_strategy_timestamp_list = []
        # closing_strategy_timestamp_list = []
        # opening_count = [0]*5
        # closing_count = [0]*5
        # opening_amount = [0]*5
        # closing_amount = [0]*5

        # for stack_order in strategy:
        #     if stack_order["action"] == "buy":
        #         order_size = 0
        #         for order in stack_order["order"]:
        #             order_size += order["amount"]
        #         if order_size == stack_order["position"]:
        #             print(stack_order["timestamp"])
        #             opening_strategy_timestamp_list.append(stack_order["timestamp"])
        #             opening_data = data[data['timestamp'] == stack_order['timestamp']]
        #             label_index = df_label[df_label['timestamp'] == stack_order['timestamp']]['label'].values[0]    # 获取label
        #             opening_count[label_index] += 1         # 记录对应label的开仓次数
        #             opening_amount[label_index] +=  sum( level['amount'] for level in stack_order['order']) # 记录开仓总量至对应label

        #     elif stack_order["action"] == "sell":
        #         if stack_order["position"] == 0:
        #             print(stack_order["timestamp"])
        #             closing_strategy_timestamp_list.append(stack_order["timestamp"])
        #             closing_data = data[data['timestamp'] == stack_order['timestamp']]
        #             label_index = df_label[df_label['timestamp'] == stack_order['timestamp']]['label'].values[0]
        #             closing_count[label_index] += 1
        #             closing_amount[label_index] +=  sum( level['amount'] for level in stack_order['order'])

        # print("opening_count", opening_count)
        # print("closing_count",closing_count)
        # print("opening_amount",opening_amount)
        # print("closing_amount",closing_amount)

        # opening_count_perc = [ x/sum(opening_count) for x in opening_count]
        # closing_count_perc = [ x/sum(closing_count) for x in closing_count]
        # opening_amount_perc = [ x/sum(opening_amount) for x in opening_amount]
        # closing_amount_perc = [ x/sum(closing_amount) for x in closing_amount]

        # print("opening_count_perc",opening_count_perc)
        # print("closing_count_perc",closing_count_perc)
        # print("opening_amount_perc",opening_amount_perc)
        # print("closing_amount_perc",closing_amount_perc)

        # return opening_count, closing_count, opening_amount, closing_amount

    def analysis_along_time_dynamics(
        self, path, num_dynamics, num_seg, selected_timestamp: list
    ):
        # TODO 给定段时间，先把这段时间内的市场状态划分成num_dynamics个市场状态
        # TODO 根据策略 用开平仓的次数来划分时间（总开平仓次数）
        # TODO 两维度从时间和市场状态 计算胜率，收益率，持仓时间，最大回撤，calmar ratio，已经开仓量和平仓量占总开仓量和平仓量的百分比

        selected_market = data[
            (data["timestamp"] >= selected_timestamp[0])
            & (data["timestamp"] <= selected_timestamp[1])
        ]

        model = Linear_Market_Dynamics_Model(
            data=selected_market.reset_index(), dynamic_number=num_dynamics
        )
        model.run(path)
        df_label = pd.read_feather(path + "/df_label.feather")
        assert len(selected_market) == len(df_label)

        opening_count_seg = [[0] * num_dynamics] * num_seg
        closing_count_seg = [[0] * num_dynamics] * num_seg
        opening_amount_seg = [[0] * num_dynamics] * num_seg
        closing_amount_seg = [[0] * num_dynamics] * num_seg

        opening_strategy_timestamp_list = []
        closing_strategy_timestamp_list = []

        for stack_order in self.strategy:  # self.
            if stack_order["action"] == "buy":
                order_size = 0
                for order in stack_order["order"]:
                    order_size += order["amount"]
                if order_size == stack_order["position"]:
                    opening_strategy_timestamp_list.append(stack_order["timestamp"])
            elif stack_order["action"] == "sell":
                if stack_order["position"] == 0:
                    closing_strategy_timestamp_list.append(stack_order["timestamp"])
        assert len(opening_strategy_timestamp_list) == len(
            closing_strategy_timestamp_list
        )
        # print(len(opening_strategy_timestamp_list))
        assert len(opening_strategy_timestamp_list) >= num_seg
        num_trade_seg = int(len(opening_strategy_timestamp_list) / num_seg)

        def calculate(local_selected_timestamp):
            tr, mdd, cr = self.calculate_metric(local_selected_timestamp)
            print(
                "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                    i, tr, mdd, cr
                )
            )

        def along_dynamics(local_selected_timestamp, index_seg):
            selected_market = data[
                (data["timestamp"] >= local_selected_timestamp[0])
                & (data["timestamp"] <= local_selected_timestamp[1])
            ]
            selected_strategy = [  # 进行交易的market time
                item
                for item in strategy
                if local_selected_timestamp[0]
                <= item["timestamp"]
                <= local_selected_timestamp[1]
            ]

            opening_strategy_timestamp_list = []
            closing_strategy_timestamp_list = []
            opening_count = [0] * 5
            closing_count = [0] * 5
            opening_amount = [0] * 5
            closing_amount = [0] * 5

            for stack_order in selected_strategy:
                # print(stack_order)
                if stack_order["action"] == "buy":
                    order_size = 0
                    for order in stack_order["order"]:
                        order_size += order["amount"]
                    label_index = df_label[
                        df_label["timestamp"] == stack_order["timestamp"]
                    ]["label"].values[
                        0
                    ]  # 获取label
                    opening_amount[label_index] += sum(
                        level["amount"] for level in stack_order["order"]
                    )  # 记录开仓总量至对应label
                    if order_size == stack_order["position"]:
                        # print(stack_order["timestamp"])
                        opening_strategy_timestamp_list.append(stack_order["timestamp"])
                        opening_count[label_index] += 1  # 记录对应label的开仓次数
                        # print("add buy count {} for label {}".format(opening_count[label_index], label_index))
                    # sumtemp = sum( level['amount'] for level in stack_order['order'])
                    # print("add buy amount {} for label {}".format(sumtemp, label_index))

                elif stack_order["action"] == "sell":
                    label_index = df_label[
                        df_label["timestamp"] == stack_order["timestamp"]
                    ]["label"].values[0]
                    closing_amount[label_index] += sum(
                        level["amount"] for level in stack_order["order"]
                    )
                    if stack_order["position"] == 0:
                        # print(stack_order["timestamp"])
                        closing_strategy_timestamp_list.append(stack_order["timestamp"])
                        closing_count[label_index] += 1
                        # print("add sell count {} for label {}".format(closing_count[label_index], label_index))
                    # sumtemp = sum( level['amount'] for level in stack_order['order'])
                    # print("add sell record {} for label {}".format(sumtemp, label_index))

            opening_count_seg[index_seg] = opening_count
            closing_count_seg[index_seg] = closing_count
            opening_amount_seg[index_seg] = opening_amount
            closing_amount_seg[index_seg] = closing_amount

        for i in range(num_seg):
            # print()
            # print("this is segment {}".format(i))
            start_market_time = 0
            if i == 0:
                start_market_time = opening_strategy_timestamp_list[num_trade_seg * 0]
            else:
                for row_index, row in data.iterrows():
                    if (
                        row["timestamp"]
                        > closing_strategy_timestamp_list[num_trade_seg * i]
                    ):
                        start_market_time = row["timestamp"]
                        break

            if i == num_seg - 1:
                close_market_time = closing_strategy_timestamp_list[-1]
            else:
                close_market_time = closing_strategy_timestamp_list[
                    num_trade_seg * (i + 1)
                ]

            phase_selected_timestamp = [start_market_time, close_market_time]
            calculate(phase_selected_timestamp)
            along_dynamics(phase_selected_timestamp, i)

        opening_count_perc = [
            [float(format(x / sum(opening_count), ".2f")) for x in opening_count]
            for opening_count in opening_count_seg
        ]
        closing_count_perc = [
            [float(format(x / sum(closing_count), ".2f")) for x in closing_count]
            for closing_count in closing_count_seg
        ]
        opening_amount_perc = [
            [float(format(x / sum(opening_amount), ".2f")) for x in opening_amount]
            for opening_amount in opening_amount_seg
        ]
        closing_amount_perc = [
            [float(format(x / sum(closing_amount), ".2f")) for x in closing_amount]
            for closing_amount in closing_amount_seg
        ]

        color_list = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#BEB8DC", "#2878B5"]
        x_label = ["phase1", "phase2", "phase3", "phase4", "phase5"]
        y_label = ["Bull", "Rally", "Sideways", "Pullback", "Bear"]
        y_label.reverse()

        x_ax = np.array(range(len(x_label))) * 4

        fig = plt.figure(figsize=(24, 24))
        # plt.suptitle('Opening/Closing counts/amounts along different time with diff market type')

        patches = [
            mpatches.Patch(color=color_list[i], label="{:s}".format(y_label[i]))
            for i in range(len(color_list))
        ]

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.75])
        fig.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=5,
            fontsize="large",
        )

        plt.subplot(221)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Opening count")
        for i in range(5):
            plt.bar(
                [m + i * 0.75 for m in x_ax],
                [count[i] for count in opening_count_seg],
                label=y_label[i],
                width=0.75,
                color=color_list[i],
            )
        plt.xticks([i + 1 for i in x_ax], x_label, fontsize="x-large")
        for i in range(len(opening_count_seg)):
            plotdata = [count[i] for count in opening_count_seg]
            for x, y in enumerate(plotdata):
                plt.text(
                    x * 4 + i * 0.75,
                    y + 0.1,
                    format(opening_count_perc[x][i] * 100, ".0f") + "%",
                    ha="center",
                    fontsize=9,
                )

        plt.subplot(222)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Closing count")
        for i in range(5):
            plt.bar(
                [m + i * 0.75 for m in x_ax],
                [count[i] for count in closing_count_seg],
                label=y_label[i],
                width=0.75,
                color=color_list[i],
            )
        plt.xticks([i + 1.5 for i in x_ax], x_label, fontsize="x-large")
        for i in range(len(closing_count_seg)):
            plotdata = [count[i] for count in closing_count_seg]
            for x, y in enumerate(plotdata):
                plt.text(
                    x * 4 + i * 0.75,
                    y + 0.1,
                    format(closing_count_perc[x][i] * 100, ".0f") + "%",
                    ha="center",
                    fontsize=9,
                )

        plt.subplot(223)
        plt.xlabel("Label")
        plt.ylabel("Amount")
        plt.title("Opening amount")
        for i in range(5):
            plt.bar(
                [m + i * 0.75 for m in x_ax],
                [count[i] for count in opening_amount_seg],
                label=y_label[i],
                width=0.75,
                color=color_list[i],
            )
        plt.xticks([i + 1.5 for i in x_ax], x_label, fontsize="x-large")
        for i in range(len(opening_amount_seg)):
            plotdata = [count[i] for count in opening_amount_seg]
            for x, y in enumerate(plotdata):
                plt.text(
                    x * 4 + i * 0.75,
                    y + 0.1,
                    format(opening_amount_perc[x][i] * 100, ".0f") + "%",
                    ha="center",
                    fontsize=9,
                )

        plt.subplot(224)
        plt.xlabel("Label")
        plt.ylabel("Amount")
        plt.title("Closing amount")
        for i in range(5):
            plt.bar(
                [m + i * 0.75 for m in x_ax],
                [count[i] for count in closing_amount_seg],
                label=y_label[i],
                width=0.75,
                color=color_list[i],
            )
        plt.xticks([i + 1.5 for i in x_ax], x_label, fontsize="x-large")
        for i in range(len(closing_amount_seg)):
            plotdata = [count[i] for count in closing_amount_seg]
            for x, y in enumerate(plotdata):
                plt.text(
                    x * 4 + i * 0.75,
                    y + 0.1,
                    format(closing_amount_perc[x][i] * 100, ".0f") + "%",
                    ha="center",
                    fontsize=9,
                )

        img_path = path + "/time_dynamics_proportion.pdf"
        plt.savefig(img_path, bbox_inches="tight")
        # plt.show()

        return (
            opening_count_seg,
            closing_count_seg,
            opening_amount_seg,
            closing_amount_seg,
        )


if __name__ == "__main__":
    positions = np.load("data/micro_action.npy")
    data = pd.read_feather("data/test.feather")

    print("execution start")

    num_seg = 5
    num_dynamics = 5
    path = "data/formal"
    selected_timestamp = [
        pd.Timestamp("2022-08-06 19:24:12"),
        pd.Timestamp("2022-08-15 23:59:59"),
    ]

    print("flag 1")

    strategy = transform_market_order_strategy(data, positions, max_holding_number=4000)

    print("flag 2")

    analyzer = Analyzer(data, strategy)

    print("flag 3")

    time_analysis = analyzer.analysis_along_time(2)

    mean_return_rate, mean_duration, mean_mdd, win_rate = analyzer.analysis_behavior(
        analyzer.strategy
    )  #    assert len(trade_position_record) == len(corresponding_market_information)   AssertionError

    # print("flag 4")

    # opening_count_seg,closing_count_seg,opening_amount_seg,closing_amount_seg = analyzer.analysis_along_time_dynamics(path,num_dynamics,num_seg,selected_timestamp)

    # # print("mean_return_rate", mean_return_rate)
    # # print("mean_duration", mean_duration)
    # # print("mean_mdd", mean_mdd)
    # # print("win_rate", win_rate)
    # print("opening_count_seg", opening_count_seg)
    # print("closing_count_seg", closing_count_seg)
    # print("opening_amount_seg", opening_amount_seg)
    # print("closing_amount_seg", closing_amount_seg)
