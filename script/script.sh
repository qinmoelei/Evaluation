CUDA_VISIBLE_DEVICE=0 nohup python comprehensive_evaluation/analyzer1027.py  >log/analysis_assert_error_BTCT.log 2>&1 &

CUDA_VISIBLE_DEVICE=1 nohup python comprehensive_evaluation/analyzer1027.py \
  --positions_loc best_result/ETH/micro_action.npy --data_loc best_result/ETH/test.feather --path best_result/ETH/data --max_holding_number1 0.1 --commission_fee 0.00015 \
  >log/analysis_assert_error_ETH.log 2>&1 &

CUDA_VISIBLE_DEVICE=2 nohup python comprehensive_evaluation/analyzer1027.py \
  --positions_loc best_result/BTCU/micro_action.npy --data_loc best_result/BTCU/test.feather --path best_result/BTCU/data --max_holding_number1 0.01 --commission_fee 0.00015 \
  >log/analysis_assert_error_BTCU.log 2>&1 &

CUDA_VISIBLE_DEVICE=3 nohup python comprehensive_evaluation/analyzer1027.py \
  --positions_loc best_result/GALA/micro_action.npy --data_loc best_result/GALA/test.feather --path best_result/GALA/data --max_holding_number1 4000 --commission_fee 0.00015 \ 
  >log/analysis_assert_error_GALA.log 2>&1 &


