"""
bot_runner.py

An example HTTP service (FastAPI) to launch and manage bot instances and user sessions.

The API exposes two main routes:
- /ready: Healthchecks our configuration and environment variables
- /start_bot: Starts a new bot instance and returns a token for the user to join the session

It can optionally serve static content at the root URL (/). This is useful for
serving a client web app that interacts with the bot, such as the Pipecat Test UI.

"""
from daily_helpers import create_room, get_token, check_room_url
import os
import argparse
import subprocess
import atexit
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from dotenv import load_dotenv
load_dotenv(override=True)

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}


def cleanup():
    # Clean up function, just to be extra safe
    for proc in bot_procs.values():
        proc[0].terminate()
        proc[0].wait()


atexit.register(cleanup)

# ------------ Configuration ------------ #

MAX_SESSION_TIME = 10 * 60  # 10 minutes
BOT_CAN_IDLE = True  # Does the bot leave when there are no connected peers
REQUIRED_ENV_VARS = ['OPENAI_API_KEY', 'DAILY_API_KEY', 'ELEVENLABS_API_KEY']

# Static file config
SERVE_STATIC = True
STATIC_DIR = "web-ui/dist"
STATIC_ROUTE = "/static"
STATIC_INDEX = "index.html"

# Client UI config
USE_OPEN_MIC = True  # Can the user freely talk, or do they need to wait their turn?
USE_VIDEO = False  # Does this bot require user video?

# ----------------- API ----------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Optionally serve built client static files
# Note: we recommend serving your client seperate from the bot
# runner for better scalability and separation of concerns
if SERVE_STATIC:
    app.mount(STATIC_ROUTE, StaticFiles(
        directory=STATIC_DIR, html=True), name="static")

    @app.get("/{path_name:path}", response_class=FileResponse)
    async def catch_all(path_name: Optional[str] = ""):
        if path_name == "":
            return FileResponse(f"{STATIC_DIR}/{STATIC_INDEX}")

        file_path = Path(STATIC_DIR) / (path_name or "")

        if file_path.is_file():
            return file_path

        html_file_path = file_path.with_suffix(".html")
        if html_file_path.is_file():
            return FileResponse(html_file_path)

        raise HTTPException(
            status_code=404, detail="Page not found")


@app.post("/ready")
async def ready() -> JSONResponse:
    """
        - Return config blog
        - Update environment variables if specified
    """
    return JSONResponse({"ready": True})


@app.post("/start_bot")
async def start_bot(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        # Is this a webhook creation request?
        if "test" in data:
            return JSONResponse({"test": True})
    except Exception:
        pass

    # Use specified room URL, or create a new one if not specified
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)

    if not room_url:
        try:
            room_url = create_room(MAX_SESSION_TIME)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Unable to provision room")
    else:
        # Check passed room URL exists
        try:
            check_room_url(room_url)
        except Exception:
            raise HTTPException(
                status_code=500, detail=f"Room not found: {room_url}")

    # Give the agent a token to join the session
    token = get_token(room_url, MAX_SESSION_TIME)

    if not room_url or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)
    try:
        proc = subprocess.Popen(
            [
                f"python3 -m bot -u {room_url} -t {token}"
            ],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")

     # Grab a token for the user to join with
    user_token = get_token(room_url, MAX_SESSION_TIME)

    return JSONResponse({
        "bot_id": proc.pid,
        "room_url": room_url,
        "token": user_token,
        "config": {"open_mic": USE_OPEN_MIC}})


# ----------------- Main ----------------- #

if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", type=str,
                        default=os.getenv("HOST", "localhost"), help="Host address")
    parser.add_argument("--port", type=int,
                        default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true",
                        default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run(
            "bot_runner:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )

    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
