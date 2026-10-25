"""Install the built wheel locally and verify it from outside the source tree."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheel = root / "outputs/dist/local_audio_studio-0.1.0-py3-none-any.whl"
    destination = root / "work/package-verification" / uuid4().hex
    destination.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for source in (root / "audio_studio").rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                relative = source.relative_to(root).as_posix()
                if archive.read(relative) != source.read_bytes():
                    raise RuntimeError(f"Wheel contains stale source: {relative}")
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--offline", "--no-deps", "--target", str(destination), str(wheel)], check=True)
    # A separate interpreter verifies imports, bundled HTML and an actual worker
    # subprocess, with cwd outside the repository and no sys.path source shortcut.
    code = r'''
import json,inspect,sys,time
from pathlib import Path
target, root = map(Path,sys.argv[1:])
sys.path.insert(0,str(target))
import audio_studio.api
from fastapi.testclient import TestClient
assert Path(inspect.getfile(audio_studio.api)).is_relative_to(target)
with TestClient(audio_studio.api.create_app(target/'data')) as client:
    assert client.get('/').status_code==200
    assert len(client.get('/api/presets').json())==8
    assert client.get('/static/covers.js').status_code==200
    assert client.get('/static/singing.js').status_code==200
    assert client.get('/static/music.js').status_code==200
    assert client.get('/api/covers/status').json()['assets']['engine']=='rvc'
    assert client.get('/api/singing/status').json()['model']['engine']=='diffsinger'
    assert client.get('/api/music/status').json()['model']['engine']=='ace_step'
    import audio_studio.vendor.rvc.models as rvc_models
    assert Path(inspect.getfile(rvc_models)).is_relative_to(target)
    app=client.app
    submitted=app.state.jobs.submit('validate',{'csv_path':str(target/'missing.csv'),'wavs_dir':str(target)})
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        job=app.state.jobs.get(submitted['id'])
        if job['status'] in {'succeeded','failed','cancelled'}: break
        time.sleep(.05)
    assert job['status']=='succeeded', job
    assert job['result']['valid'] is False
    result={'module_path':inspect.getfile(audio_studio.api),'html_status':200,'builtin_presets':8,'worker_status':job['status'],'worker_result':job['result'],'singing_routes_and_assets':True,'rvc_vendor_path':inspect.getfile(rvc_models)}
    (root/'outputs/validation/package.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS: installed wheel, bundled HTML, API and isolated worker from outside source cwd')
'''
    subprocess.run([sys.executable, "-c", code, str(destination), str(root)], cwd=root.parent,
                   env=dict(os.environ, PYTHONUTF8="1"), check=True)
    evidence = root / "outputs/validation/package.json"
    result = json.loads(evidence.read_text(encoding="utf-8"))
    result["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    evidence.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
