"""FastAPI 應用程式組態。"""
from __future__ import annotations

from fastapi import FastAPI

from ui.routing import router


def create_app(*, with_gradio: bool = False) -> FastAPI:
    """建立 FastAPI 應用，必要時掛載 Gradio 介面。"""

    app = FastAPI(title="Quant Strategy Dashboard", version="0.1.0")
    app.include_router(router)
    if with_gradio:
        from ui.gradio_app import build_interface  # 延遲載入避免啟動開銷
        import gradio as gr

        demo = build_interface()
        gr.mount_gradio_app(app, demo, path="/")
    return app


app = create_app()


__all__ = ["create_app", "app"]
