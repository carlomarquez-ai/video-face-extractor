import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import CONFIG
from helpers import is_video_file
from video_processing import process_video

def collect_videos(videos_dir: str):
    import os
    vids = []
    if os.path.isdir(videos_dir):
        for f in sorted(os.listdir(videos_dir)):
            p = os.path.join(videos_dir, f)
            if os.path.isfile(p) and is_video_file(f):
                vids.append(p)
        return vids
    if os.path.isfile(videos_dir) and is_video_file(videos_dir):
        return [videos_dir]
    return []

def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    videos = collect_videos(cfg["videos_dir"])
    if not videos:
        print(f"No videos found in {cfg['videos_dir']}")
        sys.exit(1)
    logs = []
    workers = max(1, int(cfg["WORKERS"]))
    print(f"[Parallel] Processing {len(videos)} video(s) with {workers} worker(s)\n")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_video, v, cfg): v for v in videos}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                res = fut.result()
                logs.append(res)
            except Exception as e:
                print(f"[ERROR] {os.path.basename(v)} failed: {e}")
    with open(os.path.join(cfg["output_dir"], "index.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print("\n✅ All done. Outputs in:", cfg["output_dir"])

if __name__ == "__main__":
    main()
