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

import os
import platform

from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_footer,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine_custom import Engine
from .session_store import session_store


if is_gradio_available():
    import gradio as gr


# Customized UI main page here
# Arrange & combine components
def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    ui_engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory ({hostname})", css=CSS) as demo:

        def resolve_engine(req: gr.Request) -> Engine:
            return session_store.get_or_create_engine(req.session_hash, demo_mode=demo_mode, pure_chat=False)

        def bind_engine_to_ui(task_engine: Engine):
            # Use one UI mapping template
            # TODO: change naming (just testing session name for now)
            task_engine.manager = ui_engine.manager

        # get session id
        session_id = gr.State()

        title = gr.HTML()
        subtitle = gr.HTML()

        if demo_mode:
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        # Init empty ui components, later inject with session engine
        # Headers: empty html block, filled when enging.change_lang called
        ui_engine.manager.add_elems("head", {"title": title, "subtitle": subtitle})

        # TODO: Dataset upload
        # with gr.Tab("Chat"):
        #     engine.manager.add_elems("infer", create_infer_tab(engine))

        # gpu usage bar
        ui_engine.manager.add_elems("footer", create_footer())

        # task list + model configs
        ui_engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = ui_engine.manager.get_elem_by_id("top.lang")

        # Train/Val/Infer config tabs
        with gr.Tab("Train"):
            ui_engine.manager.add_elems(
                "train",
                create_train_tab(
                    ui_engine,
                    engine_resolver=resolve_engine,
                    bind_engine_to_ui=bind_engine_to_ui,
                ),
            )
        with gr.Tab("Evaluate & Predict"):
            ui_engine.manager.add_elems("eval", create_eval_tab(ui_engine))
        with gr.Tab("Chat"):
            ui_engine.manager.add_elems("infer", create_infer_tab(ui_engine))
        if not demo_mode:
            with gr.Tab("Export"):
                ui_engine.manager.add_elems("export", create_export_tab(ui_engine))

        all_elems = ui_engine.manager.get_elem_list()

        def on_load(request: gr.Request):
            sid = request.session_hash

            task_engine = session_store.get_or_create_engine(
                sid,
                demo_mode=demo_mode,
                pure_chat=False,
            )

            print("ui_engine_id =", id(ui_engine), "sid =", sid, "task_engine_id =", id(task_engine))

            bind_engine_to_ui(task_engine)

            yield from task_engine.resume()

        demo.load(
            on_load,
            inputs=[],
            outputs=all_elems,
            concurrency_limit=None,
        )

        # def on_lang_change(lang_value, sid, request: gr.Request):
        #     sid = sid or request.session_hash
        #     engine = session_store.get_or_create_engine(sid, demo_mode=demo_mode, pure_chat=False)
        #     bind_engine_to_ui(engine)
        #     return engine.change_lang(lang_value)

        # for session check
        # def on_lang_change(lang_value, sid, request: gr.Request):
        #     sid = sid or request.session_hash
        #     task_engine = session_store.get_or_create_engine(sid, demo_mode=demo_mode, pure_chat=False)
        #     bind_engine_to_ui(task_engine)
        #     return (
        #         f"✅ ui_engine_id={ui_engine_id}\n\n"
        #         f"✅ lang=`{lang_value}`\n\n"
        #         f"✅ session_hash=`{sid}`\n\n"
        #         f"✅ task_engine_id={id(task_engine)}"
        #     )

        def on_lang_change(lang_value, request: gr.Request | None = None):
            task_engine = session_store.get_or_create_engine(
                request.session_hash, demo_mode=demo_mode, pure_chat=False
            )
            bind_engine_to_ui(task_engine)
            return task_engine.change_lang(lang_value)

        lang.change(
            on_lang_change,
            inputs=[lang],
            outputs=all_elems,
            queue=False,
        )

        lang.input(save_config, inputs=[lang], queue=False)

    return demo


# ====== Nouse: run already defined in webui_custom.py ===============#
def create_web_demo() -> "gr.Blocks":
    engine = Engine(pure_chat=True)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory Web Demo ({hostname})", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], scale=1)
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
