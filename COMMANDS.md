# Bewährte Befehle

## Motion herunterladen (HuggingFace)

```bash
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='openhe/g1-retargeted-motions', filename='kungfu_retargeted/Roundhouse_kick.pkl', repo_type='dataset', local_dir='./motions')"
```

## PKL → CSV konvertieren

```bash
uv run python -m mjlab.scripts.pkl_to_csv \
  --input-file motions/kungfu_retargeted/Roundhouse_kick.pkl \
  --output-file motions/roundhouse_kick.csv
```

## CSV → NPZ konvertieren (MuJoCo Forward Kinematics)

```bash
uv run python -m mjlab.scripts.csv_to_npz \
  --input-file motions/roundhouse_kick.csv \
  --input-fps 30 \
  --output-name roundhouse_kick \
  --device cpu
```

Erzeugt `/tmp/motion.npz`. W&B-Registry-Upload-Fehler kann ignoriert werden.

## Motion anschauen (ohne Training, Viser im Browser)

```bash
uv run python -m mjlab.scripts.play "Mjlab-Tracking-Flat-Unitree-G1" \
  --motion-file /tmp/motion.npz \
  --agent zero --no-terminations True
```

## Training starten

```bash
uv run python -m mjlab.scripts.train "Mjlab-Tracking-Flat-Unitree-G1" \
  --env.commands.motion.motion-file /tmp/motion.npz
```

## Trainierten Checkpoint anschauen

```bash
uv run python -m mjlab.scripts.play "Mjlab-Tracking-Flat-Unitree-G1" \
  --motion-file /tmp/motion.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<timestamp>/model_500.pt
```
