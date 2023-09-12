import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import pandas as pd
import argparse
from pathlib import Path
import warnings

sys.path.append(".")
from tool import market_dynamics_modeling_analysis
from tool import label_util as util


class Linear_Market_Dynamics_Model(object):
    def __init__(
        self,
        data,
        key_indicator="bid1_price",
        timestamp="index",
        tic="tic",
        labeling_method="slope",
        min_length_limit=60,
        merging_metric="DTW_distance",
        merging_threshold=0.0003,
        merging_dynamic_constraint=1,
        filter_strength=1,
        dynamic_number=5,
        max_length_expectation=3600,
    ):
        super(Linear_Market_Dynamics_Model, self).__init__()
        self.data_path = data
        self.method = "slice_and_merge"
        self.filter_strength = filter_strength
        self.dynamic_number = dynamic_number
        self.max_length_expectation = max_length_expectation
        self.key_indicator = key_indicator
        self.timestamp = timestamp
        self.tic = tic
        self.labeling_method = labeling_method
        self.min_length_limit = min_length_limit
        self.merging_metric = merging_metric
        self.merging_threshold = merging_threshold
        self.merging_dynamic_constraint = merging_dynamic_constraint

    def file_extension_selector(self, read):
        if self.data_path.endswith(".csv"):
            if read:
                return pd.read_csv
            else:
                return pd.DataFrame.to_feather
        elif self.data_path.endswith(".feather"):
            if read:
                return pd.read_feather
            else:
                return pd.DataFrame.to_feather
        else:
            raise ValueError("invalid file extension")

    def wirte_data_as_segments(self, data, process_datafile_path):
        # get file name and extension from process_datafile_path
        file_name, file_extension = os.path.splitext(process_datafile_path)

    def run(self, path):
        print("labeling start")

        raw_data = self.data_path
        raw_data[self.tic] = raw_data["symbol"]
        raw_data[self.key_indicator] = raw_data["bid1_price"]
        raw_data[self.timestamp] = raw_data.index
        
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created! {}".format(path))
        
        process_data_path = os.path.join(path, "valid_processed.feather")
        raw_data.to_feather(process_data_path)
        self.data_path = process_data_path

        worker = util.Worker(
            self.data_path,
            "slice_and_merge",
            filter_strength=self.filter_strength,
            key_indicator=self.key_indicator,
            timestamp=self.timestamp,
            tic=self.tic,
            labeling_method=self.labeling_method,
            min_length_limit=self.min_length_limit,
            merging_threshold=self.merging_threshold,
            merging_metric=self.merging_metric,
            merging_dynamic_constraint=self.merging_dynamic_constraint,
        )
        print("start fitting")
        worker.fit(
            self.dynamic_number, self.max_length_expectation, self.min_length_limit
        )
        print("finish fitting")
        worker.label(os.path.dirname(self.data_path))
        labeled_data = pd.concat([v for v in worker.data_dict.values()], axis=0)
        flie_reader = self.file_extension_selector(read=True)
        extension = self.data_path.split(".")[-1]
        data = flie_reader(self.data_path)
        if self.tic in data.columns:
            merge_keys = [self.timestamp, self.tic, self.key_indicator]
        else:
            merge_keys = [self.timestamp, self.key_indicator]
        merged_data = data.merge(
            labeled_data, how="left", on=merge_keys, suffixes=("", "_DROP")
        ).filter(regex="^(?!.*_DROP)")
        if self.labeling_method == "slope":
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling_slope"
        else:
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling"

        process_datafile_path = (
            os.path.splitext(path)[0]
            + "_labeled_"
            + self.model_id
            + "."
            + extension
        )
        # if extension == "csv":
        #     merged_data.to_csv(process_datafile_path, index=False)
        # elif extension == "feather":
        #     merged_data.to_feather(process_datafile_path)
        print("labeling done")
        #TODO 您现在需要从后面这里修改 之前我们已经把市场切分成多段并给每段上了标签 之前这里
        #TODO 是把每段数据都存成一个文件，并用他的label在当作上级文件夹的文件名 这里面你需要把这些数据
        #TODO 中对应的timestamp所属于的市场状态保存好，以便后续evaluation的时候确定买卖的时候对应时刻所属于的市场状态
        if not os.path.exists(os.path.join(path, "valid")):
            os.makedirs(os.path.join(path, "valid"))
            for i in range(self.dynamic_number):
                os.makedirs(os.path.join(path, "valid", "label_{}".format(i)))
            
            
            
        print("start save whole file !!!!!!!!!!!!!!!!!!!!!!!!")
        merged_data.reset_index(drop=True).to_feather(
            os.path.join(
                path,
                "df_label.feather",
            )
        )
        print("saving done !!!!!!!!!!!!!!!!!!!!!!!!")
                
                
        previous_label = merged_data.label[0]
        previous_start = 0
        label_counter = [0] * self.dynamic_number
        for i in range(len(merged_data)):
            if merged_data.label[i] != previous_label:
                merged_data.iloc[previous_start:i].reset_index(drop=True).to_feather(
                    os.path.join(
                        path,
                        "valid",
                        "label_{}".format(previous_label),
                        "df_{}.feather".format(label_counter[previous_label]),
                    )
                )
                label_counter[previous_label] += 1
                previous_start = i
                previous_label = merged_data.label[i]
        

        # print("plotting start")
        # # a list the path to all the modeling visulizations
        # market_dynamic_labeling_visualization_paths = worker.plot(
        #     worker.tics, self.slope_interval, output_path, self.model_id
        # )
        # print("plotting done")
        # # if self.OE_BTC == True:
        # #     os.remove('./temp/OE_BTC_processed.csv')

        # # MDM analysis
        # MDM_analysis = market_dynamics_modeling_analysis.MarketDynamicsModelingAnalysis(
        #     process_datafile_path, self.key_indicator
        # )
        # MDM_analysis.run_analysis(process_datafile_path)
        # print("Market dynamics modeling analysis done")

        # return (
        #     os.path.abspath(process_datafile_path),
        #     market_dynamic_labeling_visualization_paths,
        # )


if __name__ == "__main__":
    data = pd.read_feather("data/test.feather")
    model = Linear_Market_Dynamics_Model(data)
    model.run('data')
