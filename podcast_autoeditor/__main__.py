import logging
import os
import sys

from .cli import main

rc = main()
# The native libraries used here (ctranslate2 / onnxruntime / torch thread pools) intermittently
# abort in their static destructors at interpreter exit on macOS ("libc++abi: ... recursive_mutex
# lock failed", exit code 134) - after every output has been written.  Flush and skip teardown
# so a successful run reports success.
logging.shutdown()
sys.stdout.flush()
sys.stderr.flush()
os._exit(int(rc or 0))
