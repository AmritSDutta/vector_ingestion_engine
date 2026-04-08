# server_simple.py
import asyncio, json, uuid
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()
sessions = {}
TOOLS = {
    "weather.lookup": lambda city: {"city": city, "tempC": 29},
    "db.schema": lambda: {"tables": {"users": ["id", "name"], "orders": ["id", "amount"]}},
    "math.eval": lambda expr: eval(expr, {"__builtins__": None}, {})
}


def encode_msg(obj): return (json.dumps(obj) + "\n\n").encode("utf-8")


@app.get("/mcp")
async def get_stream():
    sid = uuid.uuid4().hex
    q = asyncio.Queue()
    sessions[sid] = q

    async def s():
        await q.put(
            {"jsonrpc": "2.0", "method": "notifications/endpoint", "params": {"endpoint": f"/mcp?session_id={sid}"}})
        await q.put({"jsonrpc": "2.0", "method": "notifications/tools",
                     "params": {"tools": [{"name": n} for n in TOOLS.keys()]}})
        try:
            while True: yield encode_msg(await q.get())
        finally:
            sessions.pop(sid, None)

    return StreamingResponse(s(), media_type="application/json", headers={"X-Session-Id": sid})


@app.post("/mcp")
async def post_message(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body.decode())
    except Exception as e:
        return Response(status_code=400, content=f"Invalid JSON: {e}")
    if payload.get("method") == "initialize":
        return Response(
        json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"ok": True}}), media_type="application/json")
    if payload.get("method") in ("callTool", "tool.invoke"):
        params = payload.get("params", {})
        name = params.get("name")
        args = params.get("args", {})
        tool = TOOLS.get(name)
        if not tool:
            return Response(
            json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "error": {"message": "Tool not found"}}),
            media_type="application/json")
        try:
            result = tool(**args) if isinstance(args, dict) else tool(*args)
            resp = {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
            for q in list(sessions.values()):
                await q.put(resp)
            return Response(json.dumps(resp), media_type="application/json")
        except Exception as ex:
            return Response(json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "error": {"message": str(ex)}}),
                            media_type="application/json")
    return Response(json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"echo": payload}}),
                    media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
