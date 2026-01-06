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

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...extras.constants import DATA_CONFIG
from ...extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


PAGE_SIZE = 2


def prev_page(page_index: int) -> int:
    return page_index - 1 if page_index > 0 else page_index


def next_page(page_index: int, total_num: int) -> int:
    return page_index + 1 if (page_index + 1) * PAGE_SIZE < total_num else page_index


def can_preview(dataset_dir: str, dataset: list) -> "gr.Button":
    r"""Check if the dataset is a local dataset."""
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception:
        return gr.Button(interactive=False)

    if len(dataset) == 0 or "file_name" not in dataset_info[dataset[0]]:
        return gr.Button(interactive=False)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if os.path.isfile(data_path) or (os.path.isdir(data_path) and os.listdir(data_path)):
        return gr.Button(interactive=True)
    else:
        return gr.Button(interactive=False)


def _load_data_file(file_path: str) -> list[Any]:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        elif file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return list(f)


def get_preview(dataset_dir: str, dataset: list, page_index: int) -> tuple[int, list, "gr.Column"]:
    r"""Get the preview samples from the dataset."""
    with open(os.path.join(dataset_dir, DATA_CONFIG), encoding="utf-8") as f:
        dataset_info = json.load(f)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if os.path.isfile(data_path):
        data = _load_data_file(data_path)
    else:
        data = []
        for file_name in os.listdir(data_path):
            data.extend(_load_data_file(os.path.join(data_path, file_name)))

    return len(data), data[PAGE_SIZE * page_index : PAGE_SIZE * (page_index + 1)], gr.Column(visible=True)


# TODO:
def upload_dataset(
    dataset_dir: str,
    file_obj,
) -> tuple["gr.Dropdown", "gr.Button"]:
    """Upload dataset function.

    file_obj: Gradio upload file object
    return: update dataset dropdown choices + change preview button status
    """
    if file_obj is None:
        return gr.Dropdown(), gr.Button()

    dataset_dir_p = Path(dataset_dir).expanduser().resolve()
    dataset_dir_p.mkdir(parents=True, exist_ok=True)

    # upload directory
    uploads_dir = dataset_dir_p / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # temp file directory
    tmp_path = Path(file_obj.name).resolve()

    # destination file name
    dst_name = tmp_path.name
    dst_path = uploads_dir / dst_name

    # if same filename, add time stamp
    if dst_path.exists():
        stem, suffix = dst_path.stem, dst_path.suffix
        dst_path = uploads_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"

    shutil.copy2(tmp_path, dst_path)

    # Update dataset_info.json
    config_path = dataset_dir_p / DATA_CONFIG
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
    except Exception:
        dataset_info = {}

    dataset_key = dst_path.stem

    rel_file = str(dst_path.relative_to(dataset_dir_p))
    dataset_info[dataset_key] = {"file_name": rel_file}

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    return gr.Dropdown(choices=sorted(dataset_info.keys()), value=dataset_key), gr.Button()


def create_preview_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> dict[str, "Component"]:
    # Min_width of column > 160, set small for smaller button
    with gr.Column(min_width=160):
        data_upload_btn = gr.UploadButton(
            "Upload dataset",
            interactive=True,
            scale=1,
            file_types=[".json", ".jsonl", ".txt"],
            file_count="single",
        )
        data_preview_btn = gr.Button(interactive=False, scale=1)

    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0)
            page_index = gr.Number(value=0, interactive=False, precision=0)

        with gr.Row():
            prev_btn = gr.Button()
            next_btn = gr.Button()
            close_btn = gr.Button()

        with gr.Row():
            preview_samples = gr.JSON()

    dataset.change(can_preview, [dataset_dir, dataset], [data_preview_btn], queue=False).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    data_preview_btn.click(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    prev_btn.click(prev_page, [page_index], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    next_btn.click(next_page, [page_index, preview_count], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    close_btn.click(lambda: gr.Column(visible=False), outputs=[preview_box], queue=False)

    data_upload_btn.upload(
        upload_dataset,
        inputs=[dataset_dir, data_upload_btn],
        outputs=[dataset, data_preview_btn],
        queue=False,
    ).then(
        can_preview,
        [dataset_dir, dataset],
        [data_preview_btn],
        queue=False,
    )

    return dict(
        data_preview_btn=data_preview_btn,
        data_upload_btn=data_upload_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        preview_samples=preview_samples,
    )
