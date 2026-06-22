import re

with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

before = content.count('use_container_width=True')
content = content.replace('use_container_width=True', "width='stretch'")
content = content.replace('use_container_width=False', "width='content'")

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Fixed {before} use_container_width occurrences')
