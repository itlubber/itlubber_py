import os
import sys
import time
import logging
import uvicorn
from pprint import pformat
from loguru import logger
from uvicorn.config import LOGGING_CONFIG
from handler import InterceptHandler
from pydantic import BaseModel
from starlette import status
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


# ------------------------------ 日志组件 ------------------------------ #
LOG_DIR = os.path.join(os.path.abspath("."), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FORMAT = "<green>{time:YYYY-mm-dd HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def format_record(record: dict) -> str:
    format_string = LOG_FORMAT
    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(record["extra"]["payload"], indent=4, compact=True, width=88)
        format_string += "\n<level>{extra[payload]}</level>"
    format_string += "{exception}\n"
    return format_string


logging.getLogger().handlers = [InterceptHandler()]
logger.configure(handlers=[{"sink": sys.stdout, "level": logging.DEBUG, "format": format_record}])
logger.add(os.path.join(LOG_DIR, "run.log"), encoding='utf-8', rotation="0:00")
logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
logging.getLogger("uvicorn.asgi").handlers = [InterceptHandler()]
LOGGING_CONFIG["loggers"]["uvicorn.access"]["propagate"] = True


# ------------------------------ 定义业务代码 ------------------------------ #
api_router = APIRouter()


class User(BaseModel):
    name: str
    age: int


@api_router.get('/user', status_code=201)
async def user(user_id: int):
    if user_id < 0:
        raise HTTPException(status_code=418, detail="user id must ge 0.")
    return {'user_id': user_id}


@api_router.post("/users/")
async def create_user(user: User):
    return user


# ------------------------------ 路由组件管理 ------------------------------ #
api_routes = APIRouter()
api_routes.include_router(api_router, prefix='/v1.0')


# ------------------------------ fastapi应用 ------------------------------ #
def create_start_app_handler():
	async def startup():
		pass

	return startup


def create_stop_app_handler():
	async def shutdown():
		pass

	return shutdown


def create_app():
    app = FastAPI(title="itlubber's fastapi cli", debug=True, version="1.0")

    # 跨域问题
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加全局注册和销毁事件
    app.add_event_handler("startup", create_start_app_handler())
    app.add_event_handler("shutdown", create_stop_app_handler())
    # 添加请求拦截
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        response = await call_next(request)
        return response

    # 路由管理, 可添加多个路由
    app.include_router(api_routes, prefix="/api")

    return app


app = create_app()


if __name__ == '__main__':
    # 命令行启动方式:  uvicorn main:app  --port 8000 --debug --reload --log-config uvicorn_config.json
    uvicorn.run(app, host="0.0.0.0", port=8000)
