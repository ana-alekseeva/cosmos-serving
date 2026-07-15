# Runbook — deploy an endpoint, then run the RoboLab eval (Job 3)

End-to-end steps to stand up a `Cosmos3-Nano-Policy-DROID` serving endpoint on Nebius and score
it with the RoboLab subset. Two phases with **different** tools:

1. **Serve** the model — `npa workbench cosmos deploy` (H100 serverless endpoint).
2. **Evaluate** it — a Nebius **AI Job** on an RT-core L40S that drives RoboLab against the
   endpoint's OpenPI websocket. **The eval job does not discover the endpoint — you must copy the
   endpoint URL and pass it in as an env var** (see Step 5).

The gate compares a **baseline (E0)** endpoint against an **optimized (E4)** endpoint, so for a
full run you deploy **two** endpoints and launch **two** eval jobs (one per endpoint).

Concrete account values used below (from `~/.npa/config.yaml`):

| Thing | Value |
|---|---|
| Project id | `project-e00em6gppr002a5efwp7eb` |
| npa project alias | `eu-north1` |
| Registry | `cr.eu-north1.nebius.cloud/e00k6drmprp0pm6zcf` |
| Serve image | `cosmos-droid-vllm:v4` (built from `deploy/Dockerfile.serve`) |

---

## 0. One-time prerequisites

```bash
uv sync
bash deploy/install_npa.sh            # install npa (editable) into the project venv
npa configure --interactive           # ~/.npa/{credentials,config}.yaml — needs the "AI Jobs" IAM role
cp .env.example .env                  # then fill it in (next line)
```

Fill `.env` (gitignored) with at least:

```bash
HF_TOKEN=hf_...                        # Cosmos3-Nano-Policy-DROID license accepted on HF
AWS_ACCESS_KEY_ID=...                  # Nebius object storage (S3) — result uploads
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
OUTPUT_URI_EVAL=s3://serverless-challenge/robolab-eval-results/subset/   # Job 3 destination (separate from Job 2 OUTPUT_URI)
```

---

## 1. Build + push the serve image (once per code/stack change)

`npa cosmos deploy --runtime serverless` runs the image's **own ENTRYPOINT** (it injects env vars,
never a command), so the image must start the server itself. `cosmos-droid-vllm:v4` does that.
**Bump the tag on every push** — the k8s node serves stale layers on a reused tag.

```bash
export HF_TOKEN=...                    # required by the base image build
REG=cr.eu-north1.nebius.cloud/e00k6drmprp0pm6zcf
docker build --platform linux/amd64 -f deploy/Dockerfile.serve -t $REG/cosmos-droid-vllm:v4 .
docker push $REG/cosmos-droid-vllm:v4
```

If you bump the tag (e.g. `:v5`), pass it to the deploy in Step 2 with `IMAGE=$REG/cosmos-droid-vllm:v5`.

---

## 2. Create the endpoint(s)

`jobs/deploy-optimized.sh` wraps `npa workbench cosmos deploy`. `MODE` selects the config
(`CONFIG=E0` baseline eager / `CONFIG=E4` FP8 final), which the image entrypoint turns into the
serve flags. Defaults: `-p eu-north1`, `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, `--auth none`,
`IMAGE=…/cosmos-droid-vllm:v4`. `--wait` blocks until the endpoint reaches RUNNING.

```bash
export HF_TOKEN=...

# Baseline (E0) -> workbench alias cosmos-policy-baseline
MODE=baseline  bash jobs/deploy-optimized.sh

# Optimized (E4) -> workbench alias cosmos-policy-optimized   (only needed for the full gate)
MODE=optimized bash jobs/deploy-optimized.sh
```

> **`--auth none` matters.** RoboLab's websocket client and the latency harness send no auth
> header; a `token` endpoint would 401 every eval request.

---

## 3. Get the endpoint URL  ← you need this for Step 5

The URL is **not** printed anywhere the eval job can read — you copy it here and pass it in later.
It looks like `https://port8080-<hash>.tunnel.applications.eu-north1.nebius.cloud` (the `port8080-`
prefix is the container's port 8080 exposed through Nebius's applications tunnel).

Find it either way:

```bash
# via the workbench alias (whatever MODE you deployed):
npa workbench cosmos -p eu-north1 -n cosmos-policy-baseline status --output json

# or list the raw serverless endpoints + their URLs:
nebius ai endpoint list --parent-id project-e00em6gppr002a5efwp7eb --format json \
  | python3 -c 'import sys,json;[print(i["metadata"]["name"], i.get("status",{}).get("url")) for i in json.load(sys.stdin)["items"]]'
```

Save them:

```bash
E0_URL=https://port8080-XXXX.tunnel.applications.eu-north1.nebius.cloud   # cosmos-policy-baseline
E4_URL=https://port8080-YYYY.tunnel.applications.eu-north1.nebius.cloud   # cosmos-policy-optimized
```

---

## 4. Wait until it is actually SERVING (RUNNING ≠ ready)

Nebius marks the endpoint `RUNNING` when the **container** starts, but the vLLM-Omni server inside
still needs ~150s to load the model **plus** a first-run ~30 GB weight download. Until it binds
`:8080`, every path returns 404/502. Wait for the probe below to pass:

```bash
curl -sf $E0_URL/health && echo OK                              # 200 -> server is up
curl -si -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  $E0_URL/v1/realtime/robot/openpi | head -1                    # HTTP/1.1 101 -> OpenPI route mounted
```

- `200` + `101` → good, proceed.
- `404`/`502` → still loading; watch the logs and retry:
  `nebius ai endpoint logs <endpoint-id> --follow` (look for `Uvicorn running on http://0.0.0.0:8080`).

**Do not start the eval until you see `101`** — a not-ready endpoint just fails the job.

---

## 5. Start the eval job — passing the URL in

The eval runs as a Nebius AI Job on an RT-core L40S (Isaac Sim needs RT cores the H100 lacks). It
clones `cosmos-serving@main` + RoboLab, then runs `run_robolab.py`. **You must pass the endpoint
URL as an env var** — `ROLE` picks which one:

> **Platform = `gpu-l40s-a` (Intel Ice Lake L40S), not `gpu-l40s-d`.** `gpu-l40s-d` is the AMD
> Epyc L40S, which is not available to this tenant; `gpu-l40s-a` is the Intel-host variant.
> (The serving endpoints run on `gpu-h100-sxm` = Intel Sapphire Rapids, already Intel.)


| ROLE | env var you must set | side scored |
|---|---|---|
| `baseline` | `COSMOS_ENDPOINT_BASELINE=<E0_URL>` | baseline |
| `optimized` | `COSMOS_ENDPOINT_OPTIMIZED=<E4_URL>` | candidate |

`--inject-file .env:/work/.env` carries the secrets (`HF_TOKEN`, AWS keys, `OUTPUT_URI_EVAL`);
`--inject-file deploy/run_robolab_job.sh:/run_robolab_job.sh` is the entrypoint.

**Baseline job (launch once E0 shows `101`):**
```bash
nebius ai job create --parent-id project-e00em6gppr002a5efwp7eb --name robolab-baseline \
  --image nvcr.io/nvidia/isaac-lab:2.3.2 --platform gpu-l40s-a --preset 1gpu-32vcpu-128gb \
  --inject-file deploy/run_robolab_job.sh:/run_robolab_job.sh --inject-file .env:/work/.env \
  --env ROLE=baseline --env COSMOS_ENDPOINT_BASELINE=$E0_URL \
  --container-command bash --args /run_robolab_job.sh
```

**Optimized job (launch once E4 shows `101`):**
```bash
nebius ai job create --parent-id project-e00em6gppr002a5efwp7eb --name robolab-optimized \
  --image nvcr.io/nvidia/isaac-lab:2.3.2 --platform gpu-l40s-a --preset 1gpu-32vcpu-128gb \
  --inject-file deploy/run_robolab_job.sh:/run_robolab_job.sh --inject-file .env:/work/.env \
  --env ROLE=optimized --env COSMOS_ENDPOINT_OPTIMIZED=$E4_URL \
  --container-command bash --args /run_robolab_job.sh
```

The two jobs are independent and run in parallel. Records upload to
`${OUTPUT_URI_EVAL}raw/robolab/`. Per-task records resume, so a relaunch after a crash continues
rather than re-simulating the whole set.

---

## 6. Monitor

```bash
nebius ai job list --parent-id project-e00em6gppr002a5efwp7eb --format json | \
  python3 -c 'import sys,json;[print(j["metadata"]["name"], j.get("status",{}).get("state")) for j in json.load(sys.stdin)["items"]]'
nebius ai job logs <job-id> --follow
```

---

## 7. Gate (after both jobs finish)

```bash
set -a && source .env && set +a
aws s3 sync ${OUTPUT_URI_EVAL}raw/robolab/ results/robolab/ --endpoint-url "$AWS_ENDPOINT_URL"
uv run python run_robolab.py --backend vllm --side both \
  --endpoint-baseline $E0_URL --endpoint-candidate $E4_URL --robolab-root RoboLab
# PASS if candidate success drop <= SUCCESS_DROP_THRESHOLD (0.03); REJECT otherwise.
```

---

## 8. Stop billing

Serverless endpoints bill while RUNNING — tear them down when done:

```bash
npa workbench cosmos -p eu-north1 -n cosmos-policy-baseline  teardown --yes
npa workbench cosmos -p eu-north1 -n cosmos-policy-optimized teardown --yes
```

---

## Troubleshooting (first-run unknowns)

| Symptom | Likely cause / fix |
|---|---|
| Deploy fails `ImageNotFound` | The `:v4` tag isn't pushed, or you passed a tag that doesn't exist. Rebuild/push (Step 1) or fix `IMAGE=`. |
| Endpoint `RUNNING` but 404/502 | Still loading the model (~150s + ~30 GB download). Wait for `/health`=200 (Step 4). |
| OpenPI probe returns `404` (not `101`) | `--stage-overrides` / `cosmos_framework` not active in the image — RoboLab would fail. Rebuild the serve image. |
| Every eval request 401s | Endpoint deployed with `--auth token`. Redeploy with `AUTH=none`. |
| Eval job can't pull `nvcr.io/nvidia/isaac-lab:2.3.2` | NGC may need auth — add `--registry-secret <ngc-secret>` to `nebius ai job create`. |
| Eval job: RoboLab clone fails | `NVLabs/RoboLab@main` access — make it reachable (public or token). |
| Eval job: `python: command not found` / RoboLab import errors | The isaac-lab image has no bare `python`; the script auto-detects `/isaac-sim/python.sh`. Override with `--env ROBOLAB_PYTHON=<path>` if needed. |
| Records landed in the Job 2 bucket | `OUTPUT_URI_EVAL` unset — set it in `.env` (defaults to the robolab bucket otherwise). |
