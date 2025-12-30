# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.misc import use_modelscope, use_openmind
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import can_quantize, can_quantize_to, check_template, get_model_info, list_checkpoints, switch_hub


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


# helper:
def fetch_training_tasks():
    # TODO: 你自己從「全域訓練隊列/資料庫/檔案/Redis」拿資料
    # 回傳格式要符合 gr.Dataframe：list[list[Any]] 或 pandas.DataFrame
    progress = random.randint(0, 100)
    return [
        ["alice", "job_001", "running", "llama3", f"{progress}%"],
        ["bob", "job_002", "queued", "qwen2", f"{progress}%"],
    ]


def create_top() -> dict[str, "Component"]:
    # change top layouts: add task list
    with gr.Row():
        task_list = gr.Dataframe(
            headers=["user", "job_id", "status", "model", "progress"],
            row_count=8,
            col_count=5,
            interactive=False,
            wrap=True,
            label="Current training tasks",
        )

        # add task list event (periodic refresh):
        task_timer = gr.Timer(2.0)
        task_timer.tick(
            fn=fetch_training_tasks,
            outputs=[task_list],
            queue=False,
        )

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], value=None, scale=1)
        available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=2)
        model_path = gr.Textbox(scale=2)
        default_hub = "modelscope" if use_modelscope() else "openmind" if use_openmind() else "huggingface"
        hub_name = gr.Dropdown(choices=["huggingface", "modelscope", "openmind"], value=default_hub, scale=2)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=6)

    with gr.Row():
        quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", allow_custom_value=True)
        quantization_method = gr.Dropdown(choices=["bnb", "hqq", "eetq"], value="bnb")
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default")
        rope_scaling = gr.Dropdown(choices=["none", "linear", "dynamic", "yarn", "llama3"], value="none")
        booster = gr.Dropdown(choices=["auto", "flashattn2", "unsloth", "liger_kernel"], value="auto")

    # submit events
    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    ).then(check_template, [lang, template])
    model_name.input(save_config, inputs=[lang, hub_name, model_name], queue=False)
    model_path.input(save_config, inputs=[lang, hub_name, model_name, model_path], queue=False)

    finetuning_type.change(can_quantize, [finetuning_type], [quantization_bit], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    quantization_method.change(can_quantize_to, [quantization_method], [quantization_bit], queue=False)
    hub_name.change(switch_hub, inputs=[hub_name], queue=False).then(
        get_model_info, [model_name], [model_path, template], queue=False
    ).then(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False).then(
        check_template, [lang, template]
    )
    hub_name.input(save_config, inputs=[lang, hub_name], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        hub_name=hub_name,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        quantization_method=quantization_method,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
        task_list=task_list,
    )
