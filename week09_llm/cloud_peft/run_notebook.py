"""Execute cloud notebook code in order; historical exercises remain disabled."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Require matching successful GPU smoke results before 500 updates')
    parser.add_argument('--smoke-results', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path.cwd())
    parser.add_argument('--name', default='python_custom_lora_cloud')
    args = parser.parse_args()
    if args.full and not args.smoke_results:
        parser.error('--full requires --smoke-results')
    notebook_path = Path(__file__).resolve().parents[1] / 'Practice_peft_cloud.ipynb'
    smoke_path = args.smoke_results.resolve() if args.smoke_results else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_root)
    os.environ.setdefault('HF_HOME', str(Path.cwd() / 'hf-cache'))
    notebook = json.loads(notebook_path.read_text())
    scope = {'__name__': '__main__'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        print('EXECUTING', cell['id'], flush=True)
        exec(compile(''.join(cell['source']), f"{notebook_path.name}:{cell['id']}", 'exec'), scope)
        if cell['id'] == 'execution-controls':
            scope.update(RUN_LEGACY_EXPERIMENTS=False, RUN_FINAL_EXPERIMENT=True,
                         SMOKE_ONLY=not args.full, EXPERIMENT_NAME=args.name)
            if smoke_path:
                scope['SMOKE_RESULTS_PATH'] = str(smoke_path)


if __name__ == '__main__':
    main()
