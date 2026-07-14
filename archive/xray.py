# xray.py

import json
import time
from pathlib import Path
from dataclasses import asdict

from almond import Almond, AlmondConfig


class AlmondXRay:

    def __init__(self, almond):
        self.almond = almond

        self.report = {
            "timestamp": time.time(),
            "question": "",
            "prepare_context": {},
            "context_blocks": [],
            "messages": [],
            "llm_response": "",
            "retrieval_trace": {},
            "pool_snapshot": {},
        }

        self._install_hooks()

    # ============================================================
    # HOOKS
    # ============================================================

    def _install_hooks(self):

        self._hook_prepare_context()
        self._hook_build_messages()
        self._hook_call_llm()

    # ------------------------------------------------------------

    def _hook_prepare_context(self):

        original = self.almond.controller.prepare_context

        def wrapped(query):

            result = original(query)

            self.report["question"] = query

            try:

                trace = getattr(
                    self.almond.controller,
                    "last_retrieval_trace",
                    {}
                )

                self.report["retrieval_trace"] = trace

            except Exception as e:

                self.report["retrieval_trace_error"] = str(e)

            try:

                self.report["prepare_context"] = {
                    "returned_blocks": len(result),
                }

                self.report["context_blocks"] = [
                    {
                        "id": b.id,
                        "tier": b.tier.value,
                        "tag": b.tag.value,
                        "preview": b.content[:500]
                    }
                    for b in result
                ]

            except Exception as e:

                self.report["context_error"] = str(e)

            return result

        self.almond.controller.prepare_context = wrapped

    # ------------------------------------------------------------

    def _hook_build_messages(self):

        original = self.almond._build_messages

        def wrapped(*args, **kwargs):

            messages = original(*args, **kwargs)

            try:

                self.report["messages"] = messages

                if messages:

                    self.report["system_prompt_preview"] = (
                        messages[0]["content"][:10000]
                    )

            except Exception as e:

                self.report["message_error"] = str(e)

            return messages

        self.almond._build_messages = wrapped

    # ------------------------------------------------------------

    def _hook_call_llm(self):

        original = self.almond._call_llm

        def wrapped(messages):

            response = original(messages)

            self.report["llm_response"] = response

            return response

        self.almond._call_llm = wrapped

    # ============================================================
    # SNAPSHOTS
    # ============================================================

    def snapshot_pool(self):

        try:

            pool = self.almond.controller.dump_pool()

            self.report["pool_snapshot"] = {
                "count": len(pool),
                "l1": sum(
                    1 for x in pool
                    if x["tier"] == "L1_HOT_CACHE"
                ),
                "l2": sum(
                    1 for x in pool
                    if x["tier"] == "L2_ACTIVE_RAM"
                ),
                "l3": sum(
                    1 for x in pool
                    if x["tier"] == "L3_VIRTUAL_SWAP"
                ),
                "l4": sum(
                    1 for x in pool
                    if x["tier"] == "L4_ARCHIVE"
                ),
            }

        except Exception as e:

            self.report["pool_error"] = str(e)

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):

        Path("xray_runs").mkdir(exist_ok=True)

        filename = (
            f"xray_runs/"
            f"xray_{int(time.time())}.json"
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.report,
                f,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        print(f"\n[XRAY SAVED] {filename}")

        return filename


# ================================================================
# TEST DRIVER
# ================================================================

if __name__ == "__main__":

    config = AlmondConfig(
        benchmark_mode=True
    )

    almond = Almond(config)

    xray = AlmondXRay(almond)

    print("\n=== ALMOND XRAY ===")

    while True:

        q = input("\nQuestion > ")

        if q.lower() in {"exit", "quit"}:
            break

        xray.snapshot_pool()

        answer = almond.chat(q)

        print("\nANSWER:")
        print(answer)

        xray.save()

    almond.close()