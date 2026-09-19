import threading
import queue
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IngestionTask:
    user_message: str
    assistant_reply: str
    store_user_message: bool
    latency_ms: float

class MemoryWorker:
    def __init__(self, almond: "Almond", maxsize: int = 100):
        self.almond = almond
        self.queue = queue.Queue(maxsize=maxsize)
        self.shutdown_event = threading.Event()
        self.worker_thread = None

    def start(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
            
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="MemoryWorker")
        self.worker_thread.start()
        logger.info("[MemoryWorker] Started.")

    def stop(self):
        if self.worker_thread is not None:
            self.shutdown_event.set()
            # Push a sentinel to unblock queue.get()
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=5.0)
            logger.info("[MemoryWorker] Stopped.")

    def enqueue(self, task: IngestionTask):
        try:
            self.queue.put_nowait(task)
            logger.info("[MemoryWorker] Enqueued task. Queue size: %d", self.queue.qsize())
        except queue.Full:
            logger.error("[MemoryWorker] QUEUE FULL (maxsize=%d). Dropping ingestion task for message: '%s'", 
                         self.queue.maxsize, task.user_message[:50])

    def _worker_loop(self):
        while not self.shutdown_event.is_set():
            try:
                task = self.queue.get(timeout=1.0)
                if task is None:
                    continue  # Sentinel for shutdown
                
                logger.info("[MemoryWorker] Processing task... Queue size: %d", self.queue.qsize())
                
                # Retry logic
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        self._execute_task(task)
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning("[MemoryWorker] Task failed on attempt %d/%d: %s. Retrying...", attempt, max_retries, e)
                            time.sleep(1.0 * attempt)
                        else:
                            logger.error("[MemoryWorker] Task failed on final attempt %d: %s", attempt, e, exc_info=True)
                            
                self.queue.task_done()
                logger.info("[MemoryWorker] Done.")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("[MemoryWorker] Unhandled loop error: %s", e, exc_info=True)

    def _execute_task(self, task: IngestionTask):
        from core.almond import ConversationPipeline
        pipeline = ConversationPipeline(self.almond)
        pipeline.execute_ingest(task)
