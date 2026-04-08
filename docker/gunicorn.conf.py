import os
import multiprocessing
bind            = f"0.0.0.0:{os.getenv('PORT', '7860')}"
backlog         = int(os.getenv("UVICORN_BACKLOG", "2048"))
_env_workers = os.getenv("WORKERS", "")
workers         = int(_env_workers) if _env_workers.isdigit() else 1
worker_class    = "uvicorn.workers.UvicornWorker"
worker_connections = int(os.getenv("MAX_CONNECTIONS", "1000"))
threads         = 1
max_requests    = 1000
max_requests_jitter = 100
timeout         = 120
keepalive       = int(os.getenv("KEEPALIVE", "65"))
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
loglevel        = os.getenv("LOG_LEVEL", "info")
accesslog       = "-"
errorlog        = "-"
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" '
    'rt=%(L)s'
)
proc_name       = "emergi-env"
default_proc_name = "emergi-env"
preload_app     = True
reuse_port      = True
forwarded_allow_ips = "*"
proxy_protocol  = False
proxy_allow_from = "*"
limit_request_line   = 8192
limit_request_fields = 200
def on_starting(server):
    server.log.info("EMERGI-ENV Gunicorn master starting")
def when_ready(server):
    server.log.info(f"EMERGI-ENV ready — {workers} worker(s) on {bind}")
def worker_exit(server, worker):
    server.log.warning(f"Worker {worker.pid} exited")
def on_exit(server):
    server.log.info("EMERGI-ENV Gunicorn master shut down cleanly")