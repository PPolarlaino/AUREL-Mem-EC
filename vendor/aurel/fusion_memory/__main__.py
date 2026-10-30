import uvicorn

uvicorn.run("fusion_memory.app:create_app", factory=True, host="0.0.0.0", port=18080,
            workers=1, access_log=False)
