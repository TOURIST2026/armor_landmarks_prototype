"""Rebuild the self-contained annotation UI after editing config.json."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'templates/config.json').read_text(encoding='utf-8-sig'))
    images = ['data:image/png;base64,'+base64.b64encode((ROOT/'templates'/c['image']).read_bytes()).decode() for c in config['classes']]
    html = (ROOT/'tools/annotation_template.html').read_text(encoding='utf-8')
    html = html.replace('__CONFIG__',json.dumps(config,ensure_ascii=False)).replace('__IMAGES__',json.dumps(images))
    (ROOT/'annotate.html').write_text(html,encoding='utf-8')
    print(ROOT/'annotate.html')


if __name__ == '__main__':
    main()
