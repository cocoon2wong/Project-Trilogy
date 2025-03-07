# Project-Trilogy
📘
<!-- DO NOT CHANGE THIS LINE -->
---

## Args Used

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.

### Basic Args

- `--K_train`: type=`int`, argtype=`static`.
  The number of multiple generations when training. This arg only works for multiple-generation models. 
  The default value is `10`.
- `--K`: type=`int`, argtype=`dynamic`.
  The number of multiple generations when testing. This arg only works for multiple-generation models. 
  The default value is `20`.
- `--anntype`: type=`str`, argtype=`static`.
  Model's predicted annotation type. Can be `'coordinate'` or `'boundingbox'`. 
  The default value is `coordinate`.
- `--auto_clear`: type=`int`, argtype=`temporary`.
  Controls whether to clear all other saved weights except for the best one. It performs similarly to running `python scripts/clear.py --logs logs`. 
  The default value is `1`.
- `--batch_size` (short for `-bs`): type=`int`, argtype=`dynamic`.
  Batch size when implementation. 
  The default value is `5000`.
- `--compute_loss`: type=`int`, argtype=`temporary`.
  Controls whether to compute losses when testing. 
  The default value is `0`.
- `--compute_metrics_with_types`: type=`int`, argtype=`temporary`.
  Controls whether to compute metrics separately on different kinds of agents. 
  The default value is `0`.
- `--dataset`: type=`str`, argtype=`static`.
  Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. NOTE: DO NOT set this argument manually. 
  The default value is `Unavailable`.
- `--draw_results` (short for `-dr`): type=`str`, argtype=`temporary`.
  Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the `plist` file (saved in `dataset_configs` folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`. 
  The default value is `null`.
- `--draw_videos`: type=`str`, argtype=`temporary`.
  Controls whether to draw visualized results on video frames and save them as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the `plist` file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_videos` if `draw_videos != 'null'`. 
  The default value is `null`.
- `--epochs`: type=`int`, argtype=`static`.
  Maximum training epochs. 
  The default value is `500`.
- `--experimental`: type=`bool`, argtype=`temporary`.
  NOTE: It is only used for code tests. 
  The default value is `False`.
- `--feature_dim`: type=`int`, argtype=`static`.
  Feature dimensions that are used in most layers. 
  The default value is `128`.
- `--force_anntype`: type=`str`, argtype=`temporary`.
  Assign the prediction type. It is now only used for silverballers models that are trained with annotation type `coordinate` but to be tested on datasets with annotation type `boundingbox`. 
  The default value is `null`.
- `--force_clip`: type=`str`, argtype=`temporary`.
  Force test video clip (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_dataset`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_split`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--gpu`: type=`str`, argtype=`temporary`.
  Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`. NOTE: It only supports training or testing on one GPU. 
  The default value is `0`.
- `--help` (short for `-h`): type=`str`, argtype=`temporary`.
  Print help information on the screen. 
  The default value is `null`.
- `--input_pred_steps`: type=`str`, argtype=`static`.
  Indices of future time steps that are used as extra model inputs. It accepts a string that contains several integer numbers separated with `'_'`. For example, `'3_6_9'`. It will take the corresponding ground truth points as the input when training the model, and take the first output of the former network as this input when testing the model. Set it to `'null'` to disable these extra model inputs. 
  The default value is `null`.
- `--interval`: type=`float`, argtype=`static`.
  Time interval of each sampled trajectory point. 
  The default value is `0.4`.
- `--load` (short for `-l`): type=`str`, argtype=`temporary`.
  Folder to load model (to test). If set to `null`, the training manager will start training new models according to other given args. 
  The default value is `null`.
- `--log_dir`: type=`str`, argtype=`static`.
  Folder to save training logs and model weights. Logs will save at `args.save_base_dir/current_model`. DO NOT change this arg manually. (You can still change the path by passing the `save_base_dir` arg.) 
  The default value is `Unavailable`.
- `--loss_weights`: type=`str`, argtype=`dynamic`.
  Configure the agent-wise loss weights. It now only supports the dataset-clip-wise re-weight. 
  The default value is `{}`.
- `--lr` (short for `-lr`): type=`float`, argtype=`static`.
  Learning rate. 
  The default value is `0.001`.
- `--macos`: type=`int`, argtype=`temporary`.
  (Experimental) Choose whether to enable the `MPS (Metal Performance Shaders)` on Apple platforms (instead of running on CPUs). 
  The default value is `0`.
- `--max_agents`: type=`int`, argtype=`static`.
  Max number of agents to predict per frame. It only works when `model_type == 'frame-based'`. 
  The default value is `50`.
- `--model_name`: type=`str`, argtype=`static`.
  Customized model name. 
  The default value is `model`.
- `--model_type`: type=`str`, argtype=`static`.
  Model type. It can be `'agent-based'` or `'frame-based'`. 
  The default value is `agent-based`.
- `--model`: type=`str`, argtype=`static`.
  The model type used to train or test. 
  The default value is `none`.
- `--noise_depth`: type=`int`, argtype=`static`.
  Depth of the random noise vector. 
  The default value is `16`.
- `--obs_frames` (short for `-obs`): type=`int`, argtype=`static`.
  Observation frames for prediction. 
  The default value is `8`.
- `--output_pred_steps`: type=`str`, argtype=`static`.
  Indices of future time steps to be predicted. It accepts a string that contains several integer numbers separated with `'_'`. For example, `'3_6_9'`. Set it to `'all'` to predict points among all future steps. 
  The default value is `all`.
- `--pmove`: type=`int`, argtype=`static`.
  (Pre/post-process Arg) Index of the reference point when moving trajectories. 
  The default value is `-1`.
- `--pred_frames` (short for `-pred`): type=`int`, argtype=`static`.
  Prediction frames. 
  The default value is `12`.
- `--preprocess`: type=`str`, argtype=`static`.
  Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like `'111'`): - The first bit: `MOVE` trajectories to (0, 0); - The second bit: re-`SCALE` trajectories; - The third bit: `ROTATE` trajectories. 
  The default value is `100`.
- `--restore_args`: type=`str`, argtype=`temporary`.
  Path to restore the reference args before training. It will not restore any args if `args.restore_args == 'null'`. 
  The default value is `null`.
- `--restore`: type=`str`, argtype=`temporary`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`. 
  The default value is `null`.
- `--save_base_dir`: type=`str`, argtype=`static`.
  Base folder to save all running logs. 
  The default value is `./logs`.
- `--split` (short for `-s`): type=`str`, argtype=`static`.
  The dataset split that used to train and evaluate. 
  The default value is `zara1`.
- `--start_test_percent`: type=`float`, argtype=`temporary`.
  Set when (at which epoch) to start validation during training. The range of this arg should be `0 <= x <= 1`. Validation may start at epoch `args.epochs * args.start_test_percent`. 
  The default value is `0.0`.
- `--step`: type=`float`, argtype=`dynamic`.
  Frame interval for sampling training data. 
  The default value is `1.0`.
- `--test_mode`: type=`str`, argtype=`temporary`.
  Test settings. It can be `'one'`, `'all'`, or `'mix'`. When setting it to `one`, it will test the model on the `args.force_split` only; When setting it to `all`, it will test on each of the test datasets in `args.split`; When setting it to `mix`, it will test on all test datasets in `args.split` together. 
  The default value is `mix`.
- `--test_step`: type=`int`, argtype=`temporary`.
  Epoch interval to run validation during training. 
  The default value is `1`.
- `--update_saved_args`: type=`int`, argtype=`temporary`.
  Choose whether to update (overwrite) the saved arg files or not. 
  The default value is `0`.
- `--verbose` (short for `-v`): type=`int`, argtype=`temporary`.
  Controls whether to print verbose logs and outputs to the terminal. 
  The default value is `0`.

### Reverberation Args

- `--Kc`: type=`int`, argtype=`static`.
  The number of style channels in `Agent` model. 
  The default value is `20`.
- `--T` (short for `-T`): type=`str`, argtype=`static`.
  Transformation type used to compute trajectory spectrums on the ego agents. It could be: - `none`: no transformations - `haar`: haar wavelet transform - `db2`: DB2 wavelet transform 
  The default value is `haar`.
- `--compute_linear_base`: type=`int`, argtype=`static`.
  Whether to learn to forecast the linear base during training. 
  The default value is `1`.
- `--compute_re_bias`: type=`int`, argtype=`static`.
  Whether to learn to forecast the re-bias during training. 
  The default value is `1`.
- `--compute_self_bias`: type=`int`, argtype=`static`.
  Whether to learn to forecast the self-bias during training. 
  The default value is `1`.
- `--draw_kernels`: type=`int`, argtype=`temporary`.
  Choose whether to draw and show visualized kernels when testing. It is typically used in the playground mode. 
  The default value is `0`.
- `--full_steps`: type=`int`, argtype=`static`.
  Choose whether to compute all angle-based social partitions on all observation frames or just on a flattened feature. It may bring extra computation consumptions when this term is enabled. 
  The default value is `0`.
- `--partitions`: type=`int`, argtype=`static`.
  The number of partitions when computing the angle-based feature. 
  The default value is `-1`.
- `--test_with_linear_base`: type=`int`, argtype=`temporary`.
  Whether to forecast the linear base when *testing* the model. 
  The default value is `1`.
- `--test_with_re_bias`: type=`int`, argtype=`temporary`.
  Whether to forecast the re-bias when *testing* the model. 
  The default value is `1`.
- `--test_with_self_bias`: type=`int`, argtype=`temporary`.
  Whether to forecast the self-bias when *testing* the model. 
  The default value is `1`.

### Visualization Args

- `--distribution_steps`: type=`str`, argtype=`temporary`.
  Controls which time step(s) should be considered when visualizing the distribution of forecasted trajectories. It accepts one or more integer numbers (started with 0) split by `'_'`. For example, `'4_8_11'`. Set it to `'all'` to show the distribution of all predictions. 
  The default value is `all`.
- `--draw_distribution` (short for `-dd`): type=`int`, argtype=`temporary`.
  Controls whether to draw distributions of predictions instead of points. If `draw_distribution == 0`, it will draw results as normal coordinates; If `draw_distribution == 1`, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors. 
  The default value is `0`.
- `--draw_exclude_type`: type=`str`, argtype=`temporary`.
  Draw visualized results of agents except for user-assigned types. If the assigned types are `"Biker_Cart"` and the `draw_results` or `draw_videos` is not `"null"`, it will draw results of all types of agents except "Biker" and "Cart". It supports partial match, and it is case-sensitive. 
  The default value is `null`.
- `--draw_extra_outputs`: type=`int`, argtype=`temporary`.
  Choose whether to draw (put text) extra model outputs on the visualized images. 
  The default value is `0`.
- `--draw_full_neighbors`: type=`int`, argtype=`temporary`.
  Choose whether to draw the full observed trajectories of all neighbor agents or only the last trajectory point at the current observation moment. 
  The default value is `0`.
- `--draw_index`: type=`str`, argtype=`temporary`.
  Indexes of test agents to visualize. Numbers are split with `_`. For example, `'123_456_789'`. 
  The default value is `all`.
- `--draw_lines`: type=`int`, argtype=`temporary`.
  Choose whether to draw lines between each two 2D trajectory points. 
  The default value is `0`.
- `--draw_on_empty_canvas`: type=`int`, argtype=`temporary`.
  Controls whether to draw visualized results on the empty canvas instead of the actual video. 
  The default value is `0`.

### Playground Args

- `--clip`: type=`str`, argtype=`temporary`.
  The video clip to run this playground. 
  The default value is `zara1`.
- `--compute_social_diff`: type=`int`, argtype=`temporary`.
 (Working in process)
  The default value is `0`.
- `--do_not_draw_neighbors`: type=`int`, argtype=`temporary`.
  Choose whether to draw neighboring-agents' trajectories. 
  The default value is `0`.
- `--draw_seg_map`: type=`int`, argtype=`temporary`.
  Choose whether to draw segmentation maps on the canvas. 
  The default value is `1`.
- `--lite`: type=`int`, argtype=`temporary`.
  Choose whether to show the lite version of tk window. 
  The default value is `0`.
- `--physical_manual_neighbor_mode`: type=`float`, argtype=`temporary`.
  Mode for the manual neighbor on segmentation maps. - Mode `1`: Add obstacles to the given position; - Mode `0`: Set areas to be walkable. 
  The default value is `1.0`.
- `--points`: type=`int`, argtype=`temporary`.
  The number of points to simulate the trajectory of manual neighbor. It only accepts `2` or `3`. 
  The default value is `2`.
- `--save_full_outputs`: type=`int`, argtype=`temporary`.
  Choose whether to save all outputs as images. 
  The default value is `0`.
- `--weight` (short for `-w`): type=`str`, argtype=`temporary`.
  The default weights to load. 
  The default value is `static`.
<!-- DO NOT CHANGE THIS LINE -->
