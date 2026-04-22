#!/usr/bin/env python3
"""
Genera traducciones (NO/EN/IT) de todas las descripciones de rutas
y las inyecta en vinterturer_arome.html.

Uso:
  export ANTHROPIC_API_KEY=sk-ant-...
  python fetch_translations.py

O pasar la key como argumento:
  python fetch_translations.py sk-ant-...
"""
import json, re, time, sys, os

try:
    import anthropic
except ImportError:
    print("Instalando anthropic...")
    os.system("pip install anthropic -q")
    import anthropic

# API key
api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("ERROR: Falta ANTHROPIC_API_KEY")
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)

HTML_FILE = "test_v3.html"

with open(HTML_FILE, 'r') as f:
    html = f.read()

m = re.search(r'const ROUTES = (\[.*?\]);', html, re.DOTALL)
routes = json.loads(m.group(1))

LANGS = [('summary_no','Norwegian Bokmål'), ('summary_en','English'), ('summary_it','Italian')]

def translate(text, lang_name):
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role":"user","content":
            f"Translate this Spanish mountain route description to {lang_name}. "
            f"Return ONLY the translated text, no preamble.\n\n{text}"
        }]
    )
    return msg.content[0].text.strip()

total = sum(1 for r in routes if r.get('summary_es'))
done = 0

for r in routes:
    if not r.get('summary_es'):
        r.setdefault('summary_no', None)
        r.setdefault('summary_en', None)
        r.setdefault('summary_it', None)
        continue

    # Skip already translated
    if r.get('summary_en'):
        r.setdefault('summary_no', None)
        r.setdefault('summary_it', None)
        done += 1
        continue

    done += 1
    print(f"[{done}/{total}] {r['name'][:50]}", end='', flush=True)
    for key, lang in LANGS:
        try:
            r[key] = translate(r['summary_es'], lang)
            print(" ✓", end='', flush=True)
        except Exception as e:
            r[key] = None
            print(f" ✗({e})", end='', flush=True)
        time.sleep(0.2)
    print()

# Inyectar en el HTML
routes_json = json.dumps(routes, ensure_ascii=False, separators=(',',':'))
html = re.sub(r'const ROUTES = \[.*?\];', f'const ROUTES = {routes_json};', html, flags=re.DOTALL)

with open(HTML_FILE, 'w') as f:
    f.write(html)

translated = sum(1 for r in routes if r.get('summary_en'))
print(f"\n✓ {translated}/{total} rutas traducidas e inyectadas en {HTML_FILE}")
