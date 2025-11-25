fastmcp run app.py:mcp --transport streamable-http --port 8000

# config.json
{
	"name": "argo-mcp-server",
	"type": "stdio",
	"description": "argo workflow mcp server",
	"isActive": true,
	"url": "http://localhost:8000/mcp"
}

{
	"name": "argo-mcp-server",
	"type": "stdio",
	"description": "argo workflow mcp server",
	"isActive": true,
	"command": "/Users/ppalmes/phome/Research-2025/AIC-8116/mcp-argo-workflow/argomcp/.venv/bin/python",
	"args": ["/Users/ppalmes/phome/Research-2025/AIC-8116/mcp-argo-workflow/argomcp/app.py"]
}

uvicorn app:main_app --host 0.0.0.0 --port 8000
uvicorn argo-k8s:main_app --host 0.0.0.0 --port 8000
