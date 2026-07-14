"""Mock backend — everything that runs WITHOUT a GPU or the real model.

The whole harness (logs, waterfalls, aggregation, quality gate) runs end-to-end on this
package so the plumbing is validated before touching a GPU. None of it runs on the real
vLLM / vLLM-Omni path; swap `--backend mock` -> `--backend vllm` to leave it behind.

  * replay.py  — synthetic replay-set generator (produces policy/mock/manifest.json)
  * engine.py  — MockPolicyEngine: modeled per-stage latency (§6/§7)
  * robolab.py — modeled RoboLab task success (the real gate runs Isaac Sim, §4 Job 3)
"""
