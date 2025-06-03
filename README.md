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


<details>
    <summary>
        <code>--K</code>
    </summary>
    <p>
        The number of multiple generations when testing. This arg only works for multiple-generation models.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>20</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--K_train</code>
    </summary>
    <p>
        The number of multiple generations when training. This arg only works for multiple-generation models.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>10</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--anntype</code>
    </summary>
    <p>
        Model's predicted annotation type. Can be <code>'coordinate'</code> or <code>'boundingbox'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>coordinate</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--auto_clear</code>
    </summary>
    <p>
        Controls whether to clear all other saved weights except for the best one. It performs similarly to running <code>python scripts/clear.py --logs logs</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--batch_size</code> (short for <code>-bs</code>)
    </summary>
    <p>
        Batch size when implementation.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>5000</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_loss</code>
    </summary>
    <p>
        Controls whether to compute losses when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_metrics_with_types</code>
    </summary>
    <p>
        Controls whether to compute metrics separately on different kinds of agents.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--dataset</code>
    </summary>
    <p>
        Name of the video dataset to train or evaluate. For example, <code>'ETH-UCY'</code> or <code>'SDD'</code>. NOTE: DO NOT set this argument manually.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>Unavailable</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--down_sampling_rate</code>
    </summary>
    <p>
        Selects whether to down-sample from multiple-generated predicted trajectories. This arg only works for multiple-generative models.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_results</code> (short for <code>-dr</code>)
    </summary>
    <p>
        Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the <code>plist</code> file (saved in <code>dataset_configs</code> folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that <code>test_mode</code> will be set to <code>'one'</code> and <code>force_split</code> will be set to <code>draw_results</code> if <code>draw_results != 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_videos</code>
    </summary>
    <p>
        Controls whether to draw visualized results on video frames and save them as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the <code>plist</code> file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that <code>test_mode</code> will be set to <code>'one'</code> and <code>force_split</code> will be set to <code>draw_videos</code> if <code>draw_videos != 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--epochs</code>
    </summary>
    <p>
        Maximum training epochs.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>500</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--experimental</code>
    </summary>
    <p>
        NOTE: It is only used for code tests.
    </p>
    <ul>
        <li>Type=<code>bool</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>False</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--feature_dim</code>
    </summary>
    <p>
        Feature dimensions that are used in most layers.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>128</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_anntype</code>
    </summary>
    <p>
        Assign the prediction type. It is now only used for silverballers models that are trained with annotation type <code>coordinate</code> but to be tested on datasets with annotation type <code>boundingbox</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_clip</code>
    </summary>
    <p>
        Force test video clip (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_dataset</code>
    </summary>
    <p>
        Force test dataset (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_split</code>
    </summary>
    <p>
        Force test dataset (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--gpu</code>
    </summary>
    <p>
        Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to <code>-1</code>. NOTE: It only supports training or testing on one GPU.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--help</code> (short for <code>-h</code>)
    </summary>
    <p>
        Print help information on the screen.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--input_pred_steps</code>
    </summary>
    <p>
        Indices of future time steps that are used as extra model inputs. It accepts a string that contains several integer numbers separated with <code>'_'</code>. For example, <code>'3_6_9'</code>. It will take the corresponding ground truth points as the input when training the model, and take the first output of the former network as this input when testing the model. Set it to <code>'null'</code> to disable these extra model inputs.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--interval</code>
    </summary>
    <p>
        Time interval of each sampled trajectory point.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0.4</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--load</code> (short for <code>-l</code>)
    </summary>
    <p>
        Folder to load model weights (to test). If it is set to <code>null</code>, the training manager will start training new models according to other reveived args. NOTE: Leave this arg to <code>null</code> when training new models.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--load_epoch</code>
    </summary>
    <p>
        Load model weights that is saved after specific training epochs. It will try to load the weight file in the <code>load</code> dir whose name is end with <code>_epoch${load_epoch}</code>. This arg only works when the <code>auto_clear</code> arg is disabled (by passing <code>--auto_clear 0</code> when training). Set it to <code>-1</code> to disable this function.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--load_part</code>
    </summary>
    <p>
        Choose whether to load only a part of the model weights if the <code>state_dict</code> of the saved model and the model in the code do not match. *IMPORTANT NOTE*: This arg is only used for some ablation experiments. It MAY lead to incorrect predictions or metrics.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--log_dir</code>
    </summary>
    <p>
        Folder to save training logs and model weights. Logs will save at <code>${save_base_dir}/${log_dir}</code>. DO NOT change this arg manually. (You can still change the saving path by passing the <code>save_base_dir</code> arg.).
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>Unavailable</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--loss_weights</code>
    </summary>
    <p>
        Configure the agent-wise loss weights. It now only supports the dataset-clip-wise re-weight.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>{}</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--lr</code> (short for <code>-lr</code>)
    </summary>
    <p>
        Learning rate.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0.001</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--macos</code>
    </summary>
    <p>
        (Experimental) Choose whether to enable the <code>MPS (Metal Performance Shaders)</code> on Apple platforms (instead of running on CPUs).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--max_agents</code>
    </summary>
    <p>
        Max number of agents to predict per frame. It only works when <code>model_type == 'frame-based'</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>50</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model</code>
    </summary>
    <p>
        The model type used to train or test.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>none</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model_name</code>
    </summary>
    <p>
        Customized model name.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>model</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model_type</code>
    </summary>
    <p>
        Model type. It can be <code>'agent-based'</code> or <code>'frame-based'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>agent-based</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--noise_depth</code>
    </summary>
    <p>
        Depth of the random noise vector.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--depth</code>;</li>
        <li>The default value is <code>16</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--obs_frames</code> (short for <code>-obs</code>)
    </summary>
    <p>
        Observation frames for prediction.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>8</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--output_pred_steps</code>
    </summary>
    <p>
        Indices of future time steps to be predicted. It accepts a string that contains several integer numbers separated with <code>'_'</code>. For example, <code>'3_6_9'</code>. Set it to <code>'all'</code> to predict points among all future steps.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--key_points</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--pmove</code>
    </summary>
    <p>
        (Pre/post-process Arg) Index of the reference point when moving trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--pred_frames</code> (short for <code>-pred</code>)
    </summary>
    <p>
        Prediction frames.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>12</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--preprocess</code>
    </summary>
    <p>
        Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like <code>'111'</code>): - The first bit: <code>MOVE</code> trajectories to (0, 0); - The second bit: re-<code>SCALE</code> trajectories; - The third bit: <code>ROTATE</code> trajectories.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>100</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--restore</code>
    </summary>
    <p>
        Path to restore the pre-trained weights before training. It will not restore any weights if <code>args.restore == 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--restore_args</code>
    </summary>
    <p>
        Path to restore the reference args before training. It will not restore any args if <code>args.restore_args == 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--save_base_dir</code>
    </summary>
    <p>
        Base folder to save all running logs.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>./logs</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--split</code> (short for <code>-s</code>)
    </summary>
    <p>
        The dataset split that used to train and evaluate.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>zara1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--start_test_percent</code>
    </summary>
    <p>
        Set when (at which epoch) to start validation during training. The range of this arg should be <code>0 <= x <= 1</code>. Validation may start at epoch <code>args.epochs * args.start_test_percent</code>.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--step</code>
    </summary>
    <p>
        Frame interval for sampling training data.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>1.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_mode</code>
    </summary>
    <p>
        Test settings. It can be <code>'one'</code>, <code>'all'</code>, or <code>'mix'</code>. When setting it to <code>one</code>, it will test the model on the <code>args.force_split</code> only; When setting it to <code>all</code>, it will test on each of the test datasets in <code>args.split</code>; When setting it to <code>mix</code>, it will test on all test datasets in <code>args.split</code> together.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>mix</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_step</code>
    </summary>
    <p>
        Epoch interval to run validation during training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--update_saved_args</code>
    </summary>
    <p>
        Choose whether to update (overwrite) the saved arg files or not.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--verbose</code> (short for <code>-v</code>)
    </summary>
    <p>
        Controls whether to print verbose logs and outputs to the terminal.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Reverberation Args


<details>
    <summary>
        <code>--Kc</code>
    </summary>
    <p>
        The number of generations when making predictions. It is also the channels of the generating kernel in the proposed reverberation transform.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--Kg</code>, <code>--K_g</code>;</li>
        <li>The default value is <code>20</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--T</code> (short for <code>-T</code>)
    </summary>
    <p>
        Transform type used to compute trajectory spectrums. It could be: - <code>none</code>: no transformations; - <code>haar</code>: haar wavelet transform; - <code>db2</code>: DB2 wavelet transform.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>haar</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_linear</code>
    </summary>
    <p>
        (bool) Choose whether to learn to forecast the linear trajectory during training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--compute_linear_base</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_noninteractive</code>
    </summary>
    <p>
        (bool) Choose whether to learn to forecast the non-interactive trajectory during training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--learn_self_bias</code>, <code>--compute_self_bias</code>, <code>--compute_non</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_social</code>
    </summary>
    <p>
        (bool) Choose whether to learn to forecast the social trajectory during training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--learn_re_bias</code>, <code>--compute_re_bias</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--disable_G</code>
    </summary>
    <p>
        (bool) Choose whether to disable the generating kernels when applying the reverberation transform. An MSN-like generating approach will be used if this arg is set to <code>1</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--disable_R</code>
    </summary>
    <p>
        (bool) Choose whether to disable the reverberation kernels when applying the reverberation transform. Flatten and fc layers will be used if this arg is set to <code>1</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_kernels</code>
    </summary>
    <p>
        Choose whether or in which ways to draw and show visualized kernels when testing. It accepts an int value, including <code>[0, 1, 2, 3]</code>: - <code>0</code>: Do nothing; - <code>1</code>: Only visualize the reverberation kernel; - <code>2</code>: Visualize both reverberation and generating kernels; - <code>3</code>: Visualize both kernels and their inverse kernels. This arg is typically used in the playground mode.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--encode_agent_types</code>
    </summary>
    <p>
        (bool) Choose whether to encode the type name of each agent. It is mainly used in multi-type-agent prediction scenes, providing a unique type-coding for each type of agents when encoding their trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--no_interaction</code>
    </summary>
    <p>
        (bool) Whether to forecast trajectories by considering social interactions. It will compute all social-interaction-related components on the set of empty neighbors if this args is set to <code>1</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--partitions</code>
    </summary>
    <p>
        The number of partitions when computing the angle-based feature. It is only used when modeling social interactions.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_with_linear</code>
    </summary>
    <p>
        (bool) Choose whether to ignore the linear base when forecasting. It only works when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_with_noninteractive</code>
    </summary>
    <p>
        (bool) Choose whether to ignore the self-bias when forecasting. It only works when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li><li>This arg can also be spelled as<code>--test_with_non</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_with_social</code>
    </summary>
    <p>
        (bool) Choose whether to ignore the resonance-bias when forecasting. It only works when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li><li>This arg can also be spelled as<code>--test_with_soc</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Visualization Args


<details>
    <summary>
        <code>--distribution_steps</code>
    </summary>
    <p>
        Controls which time step(s) should be considered when visualizing the distribution of forecasted trajectories. It accepts one or more integer numbers (started with 0) split by <code>'_'</code>. For example, <code>'4_8_11'</code>. Set it to <code>'all'</code> to show the distribution of all predictions.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_distribution</code> (short for <code>-dd</code>)
    </summary>
    <p>
        Controls whether to draw distributions of predictions instead of points. If <code>draw_distribution == 0</code>, it will draw results as normal coordinates; If <code>draw_distribution == 1</code>, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_exclude_type</code>
    </summary>
    <p>
        Draw visualized results of agents except for user-assigned types. If the assigned types are <code>"Biker_Cart"</code> and the <code>draw_results</code> or <code>draw_videos</code> is not <code>"null"</code>, it will draw results of all types of agents except "Biker" and "Cart". It supports partial match, and it is case-sensitive.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_extra_outputs</code>
    </summary>
    <p>
        Choose whether to draw (put text) extra model outputs on the visualized images.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_full_neighbors</code>
    </summary>
    <p>
        Choose whether to draw the full observed trajectories of all neighbor agents or only the last trajectory point at the current observation moment.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_index</code>
    </summary>
    <p>
        Indexes of test agents to visualize. Numbers are split with <code>_</code>. For example, <code>'123_456_789'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_lines</code>
    </summary>
    <p>
        Choose whether to draw lines between each two 2D trajectory points.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_on_empty_canvas</code>
    </summary>
    <p>
        Controls whether to draw visualized results on the empty canvas instead of the actual video.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Playground Args


<details>
    <summary>
        <code>--clip</code>
    </summary>
    <p>
        The video clip to run this playground.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>zara1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_social_diff</code>
    </summary>
    <p>
        (Working in process).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--do_not_draw_neighbors</code>
    </summary>
    <p>
        Choose whether to draw neighboring-agents' trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_seg_map</code>
    </summary>
    <p>
        Choose whether to draw segmentation maps on the canvas.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--lite</code>
    </summary>
    <p>
        Choose whether to show the lite version of tk window.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--physical_manual_neighbor_mode</code>
    </summary>
    <p>
        Mode for the manual neighbor on segmentation maps. - Mode <code>1</code>: Add obstacles to the given position; - Mode <code>0</code>: Set areas to be walkable.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--points</code>
    </summary>
    <p>
        The number of points to simulate the trajectory of manual neighbor. It only accepts <code>2</code> or <code>3</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>2</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--save_full_outputs</code>
    </summary>
    <p>
        Choose whether to save all outputs as images.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--show_manual_neighbor_boxes</code>
    </summary>
    <p>
        (Working in process).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>
<!-- DO NOT CHANGE THIS LINE -->
